from collections import deque
import copy
from functools import partial
import os
import time

import numpy as np
import torch

from utils.env import get_vec_normalize
from utils.experiment import evaluate
from utils.utils import update_linear_schedule, update_linear_schedule_with_warmup, compute_velocity
from wrapper.PPO import ppo_update


def collect_data(policy,
                 training_envs,
                 rollouts,
                 args,
                 reward_fn,
                 env_noise_std=0.0):
    for step in range(args.n_rollout_steps):
        # Advance the system
        with torch.no_grad():
            value, action, action_log_prob, action_mean, = policy.act(rollouts.obs[step])

        # Make the transitions stochastic by adding ~ N(0, 0.05) to actions
        noise = torch.normal(0, env_noise_std, size=action.shape).to(action.device)
        action += noise

        obs, reward, _, done, infos = training_envs.step(action)

        velocity = compute_velocity(infos, obs.device)
        reward = reward_fn(reward=reward, vel=velocity)

        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

        rollouts.insert(obs, action, action_mean, action_log_prob, value, reward, masks, bad_masks, velocity)

    return rollouts


def set_params_requires_grad(net, requires_grad):
    for param in net.parameters():
        param.requires_grad = requires_grad


def opt_var_update(lambda_var, t_var, opt_var_optim, beta, gamma, c, agent, rollouts, args):
    # Freeze network parameters
    set_params_requires_grad(agent.policy, False)

    raw_rewards = rollouts.rewards
    velocities = rollouts.velocities

    # Get the last observation
    with torch.no_grad():
        next_value = agent.policy.get_value(rollouts.obs[-1]).detach()

    for _ in range(agent.n_epochs):
        # Update the rewards based using the optimized t and lambda variables
        rollouts.rewards = risk_averse_reward(raw_rewards, velocities, beta, t_var, lambda_var, c)

        # Update the returns, i.e., GAE, using the new t and lambda values
        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

        # Compute the advantages
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        # Detach to avoid "backprop twice" error
        rollouts.rewards = rollouts.rewards.detach()
        rollouts.returns = rollouts.returns.detach()

        data_generator = rollouts.full_batch_generator(advantages)

        # Since the generator yields only one batch, you don't need a loop
        full_batch_data = next(data_generator)

        # Perform operations on the full batch
        (obs_batch, actions_batch, _, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ) = full_batch_data

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, _ = agent.policy.evaluate_actions(obs_batch, masks_batch, actions_batch)
        values = values.detach()
        action_log_probs = action_log_probs.detach()

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

        # Compute the clipped surrogate (policy) loss
        surr_1 = ratio * adv_targ
        surr_2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * adv_targ

        action_loss = -torch.min(surr_1, surr_2).mean()

        # Compute the value loss
        if agent.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-args.clip_param,
                                                                                        args.clip_param)
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()

        # Clear gradients for both optimizers
        opt_var_optim.zero_grad()
        agent.optimizer.zero_grad()

        # Backward pass and optimization step for t_var and lambda_var
        (value_loss * args.value_loss_coef + action_loss).backward()

        # Invert the gradient for t_var (gradient ascent)
        lambda_var.grad = -lambda_var.grad  # lambda cannot be negative

        # Optimizer step (updates both lambda_var and t_var)
        opt_var_optim.step()

        # Clipping to ensure non-negativity
        with torch.no_grad():
            lambda_var.clamp_(min=0)

    rollouts.after_update()

    # Unfreeze network parameters
    set_params_requires_grad(agent.policy, True)

    return rollouts


def risk_averse_reward(reward, vel, beta, t_var, lambda_var, c):
    reward = reward.to(vel.device)

    penalty = c + (t_var - 1 / beta * torch.max(torch.tensor(0), t_var + vel)) 
    scaled_penalty = lambda_var * penalty

    return reward + scaled_penalty


def learn(policy, solver, training_envs, eval_envs, rollouts, args, file_name, save_dir,
          device, num_iter=int(1e6), warmup_steps=int(1e4)):
    # Get the environment and the corresponding threshold
    training_envs, vel_threshold = training_envs

    # Initialize the arrays necessary to track the training progress
    evaluations = {'reward': [], 'cost': [], 'velocity': {'mean': [], 'max': []}, 'time_steps': []}
    opt_vars = {'lambda_var': [], 't_var': []}
    episode_rewards, max_eps_reward = deque(maxlen=10), -float('inf')
    total_steps = 0
    model_save_freq = int(0.1 * num_iter)

    # First evaluation
    obs_rms = get_vec_normalize(training_envs).obs_rms
    evaluations = evaluate(evaluations, policy, eval_envs, obs_rms, args.env_noise_std, args.seed, device)
    
    # Prepare the initial observation and rollout buffer
    obs = training_envs.reset(seed=args.seed)

    solver_rollouts, opt_var_rollouts = rollouts, copy.deepcopy(rollouts)

    solver_rollouts.obs[0].copy_(obs)
    solver_rollouts.to(device)

    opt_var_rollouts.obs[0].copy_(obs)
    opt_var_rollouts.to(device)

    start = time.time()

    # Define the dual and CV@R variables, and their optimizer
    lambda_var = torch.tensor(args.lambda_init, requires_grad=True)
    t_var = torch.tensor(args.t_init, requires_grad=True)

    opt_var_optim = torch.optim.Adam([
        {'params': lambda_var, 'lr': args.lambda_lr},
        {'params': t_var, 'lr': args.t_lr}
    ])

    for iter_i in range(num_iter):
        # Linearly decrease the learning rate per update
        if args.solver_linear_lr_decay:
            update_linear_schedule(solver.optimizer, iter_i, num_iter, args.lr)
        
        if args.opt_var_linear_lr_decay and warmup_steps:
            update_linear_schedule_with_warmup(opt_var_optim, iter_i, num_iter, initial_lr=[args.lambda_lr, args.t_lr], warmup_iter=warmup_steps)
        elif args.opt_var_linear_lr_decay:
            update_linear_schedule(opt_var_optim, iter_i, num_iter, initial_lr=[args.lambda_lr, args.t_lr])

        # Take a single parameter update on the policy using the rollout data
        policy, solver, solver_rollouts, training_envs, eval_envs, total_steps, episode_rewards, evaluations = ppo_update(
            policy,
            solver,
            training_envs,
            eval_envs,
            solver_rollouts,
            args,
            total_steps,
            episode_rewards,
            evaluations,
            file_name,
            save_dir,
            device,
            opt_vars=(lambda_var.detach(), t_var.detach()),
            reward_fn=partial(risk_averse_reward,
                              beta=args.beta,
                              t_var=t_var.detach(),
                              lambda_var=lambda_var.detach(),
                              c=vel_threshold),
            env_noise_std=args.env_noise_std,
            save_model=args.save_model)

        opt_var_rollouts = collect_data(policy, training_envs, opt_var_rollouts, args, reward_fn=partial(risk_averse_reward,
                                                                                                         beta=args.beta,
                                                                                                         t_var=t_var.detach(),
                                                                                                         lambda_var=lambda_var.detach(),
                                                                                                         c=vel_threshold,),
                                                                                                         env_noise_std=args.env_noise_std)
        opt_var_rollouts = opt_var_update(lambda_var, t_var, opt_var_optim, args.beta, args.gamma, vel_threshold, solver, opt_var_rollouts, args)

        # Save the optimization variables
        opt_vars['lambda_var'].append(lambda_var.detach().item())
        opt_vars['t_var'].append(t_var.detach().item())

        np.save(os.path.join(save_dir, f"opt_vars/lambda_var/{file_name}.npy"), opt_vars['lambda_var'])
        np.save(os.path.join(save_dir, f"opt_vars/t_var/{file_name}.npy"), opt_vars['t_var'])

        # Verbosity
        if len(episode_rewards) > 1:
            end = time.time()

            total_n_rollout_steps = (iter_i + 1) * args.num_processes * args.n_rollout_steps
            print(f"Total updates: {iter_i + 1} "
                  f"Total Time Steps: {total_n_rollout_steps} "
                  f"FPS: {int(total_n_rollout_steps / (end - start))} "
                  f"Last 10 Training Episodes Reward: mean/median {np.mean(episode_rewards):.1f}/{np.median(episode_rewards):.1f}, "
                  f"max/min {np.max(episode_rewards):.1f}/{np.min(episode_rewards):.1f}")
