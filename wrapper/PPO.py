from collections import deque
import os
import json
import time

import numpy as np
import torch

from utils.env import get_vec_normalize
from utils.experiment import evaluate
from utils.utils import update_linear_schedule, compute_velocity


def ppo_update(policy,
               solver,
               training_envs,
               eval_envs,
               rollouts,
               args,
               total_steps,
               episode_rewards,
               evaluations,
               file_name,
               save_dir,
               device,
               reward_fn=None,
               env_noise_std=0.0,):
    """
     Single PPO update over rollout steps. The backbone of the learning process.
    """
    eps_vel, eps_cost = [], []
    for step in range(args.n_rollout_steps):
        # Advance the system
        with torch.no_grad():
            value, action, action_log_prob, action_mean, = policy.act(rollouts.obs[step])

        # Make the transitions stochastic by adding ~ N(0, 0.05) to actions
        noise = torch.normal(0, env_noise_std, size=action.shape).to(action.device)
        action += noise

        # Receive the reward and observe the next state
        obs, reward, cost, done, infos = training_envs.step(action)

        # Modify the reward depending on the wrapper's requirements
        velocity = compute_velocity(infos, device)
        eps_vel.append(velocity.item())
        eps_cost.append(cost.item())
        reward = reward_fn(reward=reward, vel=velocity) if reward_fn is not None else reward

        # Track the done flag
        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])
                mean_vel, max_vel, num_violations = np.mean(eps_vel), np.max(eps_vel), np.sum(eps_cost)
                eps_vel, eps_cost = [], []

        # If done, then clean the history of observations
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])

        # Add the transition to the rollout buffer
        rollouts.insert(obs, action, action_mean, action_log_prob, value, reward, masks, bad_masks, velocity)

        # Periodically evaluate the agent in a separate evaluation environment
        # Save the evaluation rewards
        if total_steps % args.eval_freq == 0 and len(episode_rewards) > 1:
            obs_rms = get_vec_normalize(training_envs).obs_rms

            evaluations = evaluate(evaluations, policy, eval_envs, obs_rms, args.env_noise_std, args.seed, device)

            np.save(os.path.join(os.path.join(save_dir, "learning_curves"), file_name), evaluations['reward'])
            np.save(os.path.join(os.path.join(save_dir, "costs"), file_name), evaluations['cost'])

        total_steps += 1

    with torch.no_grad():
        next_value = policy.get_value(rollouts.obs[-1]).detach()

    # Compute the returns, i.e., GAE
    rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)
    # Update the solver parameters
    solver.update_parameters(rollouts)
    # Prepare for the next rollout
    rollouts.after_update()

    return policy, solver, rollouts, training_envs, eval_envs, total_steps, episode_rewards, evaluations


def learn(policy, solver, training_envs, eval_envs, rollouts, args, file_name, save_dir, device, num_iter=int(1e6), warmup_steps=None):
    # Initialize the arrays necessary to track the training progress
    evaluations = {'reward': [], 'cost': [], 'velocity': {'mean': [], 'max': []}}
    episode_rewards = deque(maxlen=10)
    total_steps = 0

    # First evaluation
    obs_rms = get_vec_normalize(training_envs).obs_rms
    evaluations = evaluate(evaluations, policy, eval_envs, obs_rms, args.env_noise_std, args.seed, device)

    # Prepare the initial observation and rollout buffer
    obs = training_envs.reset(seed=args.seed)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    start = time.time()

    for iter_i in range(num_iter):
        if not args.no_linear_lr_decay:
            # Linearly decrease the learning rate per update
            update_linear_schedule(solver.optimizer, iter_i, num_iter, args.lr)

        # Take a single parameter update on the policy using the rollout data
        policy, solver, rollouts, training_envs, eval_envs, total_steps, episode_rewards, evaluations = ppo_update(
        policy,
        solver,
        training_envs,
        eval_envs,
        rollouts,
        args,
        total_steps,
        episode_rewards,
        evaluations,
        file_name,
        save_dir,
        device,
        env_noise_std=args.env_noise_std,)

        # Verbosity
        if len(episode_rewards) > 1:
            end = time.time()

            total_n_rollout_steps = (iter_i + 1) * args.num_processes * args.n_rollout_steps
            print(f"Total updates: {iter_i + 1} "
                  f"Total Time Steps: {total_n_rollout_steps} "
                  f"FPS: {int(total_n_rollout_steps / (end - start))} "
                  f"Last 10 Training Episodes Reward: mean/median {np.mean(episode_rewards):.1f}/{np.median(episode_rewards):.1f}, "
                  f"max/min {np.max(episode_rewards):.1f}/{np.min(episode_rewards):.1f}")

    return policy, None, eval_envs
