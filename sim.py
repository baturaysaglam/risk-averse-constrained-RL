import argparse
import json

import safety_gymnasium
import torch

from utils.env import make_vec_envs, get_vec_normalize
from utils.experiment import get_velocity_threshold
from utils.utils import compute_velocity
from utils.plot import *
from model.model import Policy as PPOPolicy
from solver.PPO import PPO


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for plotting the learning curves')

    # Setup
    parser.add_argument("--solver", default="PPO", help='Solver algorithm', choices=['PPO'])
    parser.add_argument('--wrapper', default='RA_C_RL', type=str, help='Wrapper learning frameworks')
    parser.add_argument("--conf", default=7, type=int, help='Configuration number for experiment')  # currently in development - to be removed later
    parser.add_argument("--env", default="Walker2d", help='OpenAI Gym environment name')
    parser.add_argument("--version", default=1, type=int, help='OpenAI Gym environment name')
    parser.add_argument("--seed", default=10, type=int, help='The seed number')
    parser.add_argument("--n_eval", default=10, type=int, help='Number of evaluations per algorithm')
    parser.add_argument("--n_bin", default=250, type=int, help='Number of bins in the histogram per algorithm')
    parser.add_argument("--sim", action='store_true', help='Perform simulation (not available on SSH)')
    parser.add_argument('--iter', default=-1, type=int, help='The checkpoint iterations (if -1, the last checkpoint will be loaded)')
    parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)')

    args = parser.parse_args()
    args_dict = vars(parser.parse_args())

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    env_name = "Safety" + args_dict["env"] + f"Velocity-v{args_dict['version']}"
    env_threshold = get_velocity_threshold(env_name)


    if args.wrapper == "RA_C_RL":
        results_dir = f"results/{args.wrapper}/conf_{args.conf}"
    else:
        results_dir = f"results/{args.wrapper}"

    model_path = os.path.join(results_dir, env_name, "models")
    opt_vars_path = os.path.join(results_dir, env_name, "opt_vars")

    if args.iter == -1:
        # Load the last checkpoint
        max_iter = -1
        for file_name in os.listdir(model_path):
            seed_num = int(file_name.split("_")[-2])
            iter_i = int(file_name.split("_")[-1].split(".")[0])
            if iter_i > max_iter and seed_num == args.seed:
                max_iter = iter_i
        iter_i = max_iter
    else:
        iter_i = args.iter

    with open(os.path.join(results_dir, "parameters.json"), 'r') as file:
        params = json.load(file)

    # env = safety_gymnasium.make(env_name)
    env = safety_gymnasium.make(env_name, render_mode='human') if args.sim else safety_gymnasium.make(env_name)
    vec_env = make_vec_envs(env_name, args.seed, 1, params["gamma"], device, False)
    vec_norm = get_vec_normalize(vec_env)

    policy = PPOPolicy(env.observation_space.shape, env.action_space, device=device)
    policy.to(device)
    agent_kwargs = {
        "policy": policy,
        "clip_param": params["clip_param"],
        "n_epochs": params["n_epochs"],
        "num_mini_batch": params["num_mini_batch"],
        "value_loss_coef": params["value_loss_coef"],
        "entropy_coef": params["entropy_coef"],
        "lr": params["lr"],
        "adam_eps": params["adam_eps"],
        "max_grad_norm": params["max_grad_norm"]
    }

    solver = PPO(**agent_kwargs)
    loads = solver.load_checkpoint(os.path.join(model_path, f"{args.solver}_{env_name}_{args.seed}_{iter_i}.pth"))
    obs_rms, _ = loads

    # Load the optimization variables
    lambda_init = params["lambda_init"]
    t_init = params["t_init"]

    lambda_var = np.load(os.path.join(opt_vars_path, 'lambda_var', f'{args.solver}_{env_name}_{args.seed}.npy'))[-1]
    t_var = np.load(os.path.join(opt_vars_path, 't_var', f'{args.solver}_{env_name}_{args.seed}.npy'))[-1]

    # Check what the optimization variables have become
    print(f"\n==================== Evaluation: {args.wrapper} ====================")
    print(f"lambda: {lambda_init:.4f} ---> {lambda_var:.4f} \t t: {t_init:.4f} --->  {t_var:.4f}\n")

    # Main infernce loop
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.training = False
        vec_norm.norm_reward = False
        vec_norm.obs_rms = obs_rms

    # Initialization of variables for tracking
    velocities = []
    eps_steps, eps_cost = 0, 0

    # The inference loop
    obs, info = env.reset(seed=args.seed + 100)
    terminated, truncated = False, False
    eps_i, eps_ret, eps_cost = 0, 0, 0
    total_steps, total_cost = 0, 0

    while eps_i < args.n_eval:
        obs = vec_norm.normalize_obs(obs)
        obs = torch.tensor(obs).float().unsqueeze(dim=0).to(device)  # Needed because parallelized environments use state normalization

        with torch.no_grad():
            _, act, _, _ = policy.act(obs, deterministic=True)
            act = act.cpu().data.numpy().squeeze()

        # Added the cost for Safe RL
        obs, reward, cost, terminated, truncated, info = env.step(act)
        velocity = compute_velocity([info], device).item()
        velocities.append(velocity)

        # Accumulate values
        eps_ret += reward
        eps_cost += cost
        eps_steps += 1
        total_cost += cost
        total_steps += 1

        if terminated or truncated:
            obs, info = env.reset(seed=args.seed + 100 + eps_i)
            print(f"\tEpisode {eps_i + 1} - reward: {eps_ret:.4f}, number of violations: {int(eps_cost)}/{int(eps_steps)}")

            # Accumulate episode cost and reset per-episode counters
            eps_i += 1
            eps_ret, eps_steps, eps_cost = 0, 0, 0

    env.close()

    avg_cost_per_eps = total_cost / args.n_eval
    avg_steps_per_eps = total_steps / args.n_eval

    print(f"\nAverage violations per episode: {avg_cost_per_eps:.3f}/{avg_steps_per_eps:.3f} - {avg_cost_per_eps / avg_steps_per_eps * 100:.3f}%\n")

    # Plotting the histograms
    title = fr'{args.env} - # steps: {total_steps}, $c$ = {env_threshold:.4f}'
    file_name = os.path.join(results_dir, env_name, 'figs', f'histogram_{args.seed}.png')

    constraint_histogram([velocities], [args.wrapper], title, env_threshold, n_bins=args.n_bin, colors=None, file_name=file_name)