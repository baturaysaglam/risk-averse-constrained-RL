import argparse
import json
import os
import random
import yaml

import numpy as np
import safety_gymnasium
import torch

from model.model import Policy as PPOPolicy
from solver.PPO import PPO
from utils.env import make_vec_envs, get_vec_normalize
from utils.experiment import get_save_dir, get_velocity_threshold, post_training_evaluation
from utils.storage import RolloutStorage
from utils.utils import DotDict
from wrapper import RA_C_RL
from wrapper import PPO as vanilla_PPO

# PPO tuned MuJoCo hyperparameters are imported from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO Implementation')

    # Setup
    parser.add_argument("--solver", default="PPO", help='Solver algorithm', choices=['PPO'])
    parser.add_argument("--wrapper", default="RA_C_RL", help='Wrapper learning framework', choices=['PPO', 'RA_C_RL'])  # RA_C_RL stands for: Risk-Averse Constraint RL (ours)
    parser.add_argument("--env", default="SafetyHopperVelocity-v1", help='OpenAI Gym environment name')
    parser.add_argument("--eval_freq", default=1e3, metavar='N', type=int, help='Evaluation period in number of time steps (default: 1e3)')
    parser.add_argument("--seed", default=4, type=int, help='Seed number for PyTorch, NumPy and OpenAI Gym (default: 0)')

    # Training duration - important because linear learning rate decay has huge impact on learning
    # Equals to 'num_iter' if given, or automatically computed at line 146
    parser.add_argument("--num_iter", default=None, type=float, metavar='N', help='Number of iterations (default: None)')
    parser.add_argument("--max_time_steps", default=1500, type=float, metavar='N', help='Maximum number of steps (default: 1e6)')

    # RA-C-RL parameters: Dual and CV@R
    parser.add_argument('--beta', type=float, default=6, help='V@R risk level')
    parser.add_argument('--c', type=float, default=None, help='Constraint variable. If not given, it is set to the velocity threshold.')

    parser.add_argument('--lambda_init', type=float, default=0.01, help='Dual (lambda) variable')
    parser.add_argument('--lambda_lr', type=float, default=1e-5, help='Learning rate for the dual variable')

    parser.add_argument('--t_init', type=float, default=15.0, help='CV@R variable')
    parser.add_argument('--t_lr', type=float, default=1e-6, help='Learning rate for the CV@R variable')

    parser.add_argument('--warmup_steps', type=float, default=0.1, help='Number of warmup steps for the learning rate of the optimization variables, as a fraction of total iterations')
    parser.add_argument('--env_noise_std', type=float, default=0.05, help='Standard deviation of the zero-mean Gaussian noise to make environment stochastic')

    # Parallelization and CUDA
    parser.add_argument('--num_processes', type=int, default=1, help='How many training CPU processes to use (default: 1)')
    parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)')

    args = parser.parse_args()
    args_dict = vars(parser.parse_args())

    # Read the meta parameters from the YAML file
    with open('meta_params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    args_dict.update(params['meta_params'])
    args = DotDict(args_dict)

    # Make sure to delete these since we don't treat them as a parameter
    del args_dict["solver"]
    del args_dict["env"]
    del args_dict["eval_freq"]
    del args_dict["seed"]
    del args_dict["num_iter"]
    del args_dict["max_time_steps"]
    del args_dict["gpu"]

    file_name = f"{args.solver}_{args.env}_{args.seed}"

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    save_dir = get_save_dir(args_dict, "./results", args.env, tuning=False)

    print("---------------------------------------------------------------------")
    print(f"Wrapper: {args.wrapper}, Solver: {args.solver}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------------------------------------")

    # Initialize the training and evaluation environments
    # Make sure the evaluation environment is of a different seed - we set seed + 100
    envs = make_vec_envs(args.env, args.seed, args.num_processes, args.gamma, device, False)
    eval_env = safety_gymnasium.make(args.env)
    eval_vec_env = make_vec_envs(args.env, args.seed + 100, 1, args_dict["gamma"], device, False)
    eval_vec_norm = get_vec_normalize(eval_vec_env)
    eval_envs = (eval_env, eval_vec_norm)
    vel_threshold = get_velocity_threshold(args.env) if args.c is None else args.c

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    state_dim = envs.observation_space.shape[0]
    action_dim = envs.action_space.shape[0]
    max_action = float(envs.action_space.high[0])

    policy = PPOPolicy(envs.observation_space.shape, envs.action_space, device=device)
    policy.to(device)

    agent_kwargs = {
        "policy": policy,
        "clip_param": args.clip_param,
        "n_epochs": args.n_epochs,
        "num_mini_batch": args.num_mini_batch,
        "value_loss_coef": args.value_loss_coef,
        "entropy_coef": args.entropy_coef,
        "lr": args.lr,
        "adam_eps": args.adam_eps,
        "max_grad_norm": args.max_grad_norm
    }

    rollout_kwargs = {
        "num_steps": args.n_rollout_steps,
        "num_processes": args.num_processes,
        "obs_shape": envs.observation_space.shape,
        "action_space": envs.action_space
    }

    rollouts = RolloutStorage(**rollout_kwargs)
    solver = PPO(**agent_kwargs)

    if "RA_" in args.wrapper:
        wrapper = RA_C_RL
        envs = (envs, vel_threshold)
    else:
        wrapper = vanilla_PPO

    if args.num_iter is None:
        num_iter = int(args.max_time_steps) // args.n_rollout_steps // args.num_processes
    else:
        num_iter = int(args.num_iter)

    warmup_steps = int(args.warmup_steps * num_iter) if args.warmup_steps is not None and args.wrapper == "RA_C_RL" else None

    policy, opt_vars, eval_envs = wrapper.learn(policy=policy,
                                                solver=solver,
                                                training_envs=envs,
                                                eval_envs=eval_envs,
                                                rollouts=rollouts,
                                                args=args,
                                                file_name=file_name,
                                                save_dir=save_dir,
                                                device=device,
                                                num_iter=num_iter,
                                                warmup_steps=warmup_steps)
    
    # Save the trained agent
    checkpoint_name = os.path.join(os.path.join(save_dir, "models"), f"{file_name}.pth")
    obs_rms = get_vec_normalize(envs[0]).obs_rms
    solver.save_checkpoint(checkpoint_name, obs_rms=obs_rms, opt_vars=opt_vars)

    # Post-training evaluation to check velocities of the trained agent
    velocities = post_training_evaluation(policy, eval_envs, obs_rms, args.env_noise_std, args.seed, device, num_eval_eps=100)
    np.save(os.path.join(os.path.join(save_dir, "velocities"), file_name), velocities, allow_pickle=True)

    envs[0].close()
    eval_envs[0].close()
