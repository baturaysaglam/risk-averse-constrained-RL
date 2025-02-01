import json
import os

import numpy as np
import torch


def get_velocity_threshold(env_name):
    velocity_map = {
        "SafetyHopperVelocity-v1": 0.7402,
        "SafetyAntVelocity-v1": 2.6222,
        "SafetyHumanoidVelocity-v1": 1.4149,
        "SafetyWalker2dVelocity-v1": 2.3415,
        "SafetyHalfCheetahVelocity-v1": 3.2096,
        "SafetySwimmerVelocity-v1": 0.2282
    }
    return velocity_map.get(env_name, None)


def post_training_evaluation(policy, eval_envs, obs_rms, env_noise_std, seed, device, num_eval_eps=100):
    eval_env, eval_vec_norm = eval_envs

    if eval_vec_norm is not None:
        eval_vec_norm.eval()
        eval_vec_norm.training = False
        eval_vec_norm.norm_reward = False
        eval_vec_norm.obs_rms = obs_rms

    state, info = eval_env.reset(seed=seed + 100)

    eps_i, eps_vel = 0, []
    velocities = []

    while eps_i < num_eval_eps:
        state = eval_vec_norm.normalize_obs(state)
        state = torch.tensor(state).float().unsqueeze(dim=0).to(device)  # Needed because parallelized environments use state normalization

        with torch.no_grad():
            _, action, _, _ = policy.act(state, deterministic=True)

            # Make the transitions stochastic by adding ~ N(0, 0.05) to actions
            noise = torch.normal(0, env_noise_std, size=action.shape).to(action.device)
            action += noise

            action = action.cpu().data.numpy().squeeze()

        # We only care about velocities
        state, _, _, done, truncated, info = eval_env.step(action)

        if 'y_velocity' not in info:
            vel = np.abs(info['x_velocity'])
        else:
            vel = np.sqrt(info['x_velocity'] ** 2 + info['y_velocity'] ** 2)

        eps_vel.append(vel)

        if done or truncated:
            state, _ = eval_env.reset(seed=seed + 100 + eps_i)
            velocities.append(eps_vel)
            eps_i += 1
            eps_vel = []

    return velocities


def evaluate(evaluations, policy, eval_envs, obs_rms, env_noise_std, seed, device, num_eval_eps=10):
    eval_env, eval_vec_norm = eval_envs

    if eval_vec_norm is not None:
        eval_vec_norm.eval()
        eval_vec_norm.training = False
        eval_vec_norm.norm_reward = False
        eval_vec_norm.obs_rms = obs_rms

    state, _ = eval_env.reset(seed=seed + 100)

    eps_i, eps_reward, eps_steps, eps_cost = 0, 0, 0, 0

    while eps_i < num_eval_eps:
        state = eval_vec_norm.normalize_obs(state)
        state = torch.tensor(state).float().unsqueeze(dim=0).to(device)  # Needed because parallelized environments use state normalization

        with torch.no_grad():
            _, action, _, _ = policy.act(state, deterministic=True)

            # Make the transitions stochastic by adding ~ N(0, 0.05) to actions
            noise = torch.normal(0, env_noise_std, size=action.shape).to(action.device)
            action += noise

            action = action.cpu().data.numpy().squeeze()

        # Added the cost for Safe RL
        state, reward, cost, done, truncated, _ = eval_env.step(action)

        eps_reward += reward
        eps_steps += 1
        eps_cost += cost

        if done or truncated:
            state, _ = eval_env.reset(seed=seed + 100 + eps_i)
            eps_i += 1

    mean_eval_reward = eps_reward / num_eval_eps
    mean_cost = eps_cost / num_eval_eps

    evaluations['reward'].append(mean_eval_reward)
    evaluations['cost'].append(mean_cost)

    print("--------------------------------------------------------")
    print(f"Evaluation over {num_eval_eps} episodes: Reward: {mean_eval_reward:.3f} - Cost: {mean_cost:.3f}")
    print("--------------------------------------------------------")

    return evaluations


def find_next_conf_number(results_dir):
    conf_dirs = [d for d in os.listdir(results_dir) if d.startswith('conf_')]
    conf_numbers = sorted([int(d.split('_')[1]) for d in conf_dirs])
    next_number = conf_numbers[-1] + 1 if conf_numbers else 1

    return next_number


def check_existing_configs(args, results_dir):
    for conf_dir in os.listdir(results_dir):
        conf_path = os.path.join(results_dir, conf_dir, 'parameters.json')

        if os.path.isfile(conf_path):
            with open(conf_path, 'r') as file:
                existing_args = json.load(file)

                if args == existing_args:
                    return conf_dir.split('_')[1]
    return None


def get_save_dir(args_dict, results_dir, env_name, tuning=False):
    results_dir = os.path.join(results_dir, args_dict["wrapper"])

    if not tuning:
        if not os.path.exists(results_dir + f"/{env_name}"):
            os.makedirs(results_dir + f"/{env_name}")

        # RA_C_RL is a special case where parameters change across environments
        if "RA_C_RL" in args_dict["wrapper"]:
            with open(os.path.join(results_dir, env_name, 'parameters.json'), 'w') as file:
                json.dump(args_dict, file, indent=4)
        else:
            with open(os.path.join(results_dir, 'parameters.json'), 'w') as file:
                json.dump(args_dict, file, indent=4)

        results_dir += f"/{env_name}"

        print(f"Saved baseline in {results_dir}\n")
    else:
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        matching_conf = check_existing_configs(args_dict, results_dir)

        if matching_conf:
            print(f"Found matching configuration: conf_{matching_conf}")
            results_dir += f"/conf_{matching_conf}/{env_name}"

            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
        else:
            next_conf_number = find_next_conf_number(results_dir)
            results_dir = os.path.join(results_dir, f'conf_{next_conf_number}')

            os.makedirs(results_dir + f"/{env_name}", exist_ok=True)

            with open(os.path.join(results_dir, 'parameters.json'), 'w') as file:
                json.dump(args_dict, file, indent=4)

            results_dir += f"/{env_name}"

            print(f"Saved new configuration in {results_dir}\n")

    curves_path = os.path.join(results_dir, "learning_curves")
    velocities_path = os.path.join(results_dir, "velocities")
    costs_path = os.path.join(results_dir, "costs")
    checkpoint_path = os.path.join(results_dir, "models")
    time_steps_path = os.path.join(results_dir, "time_steps")
    opt_vars_path = os.path.join(results_dir, "opt_vars")

    os.makedirs(curves_path, exist_ok=True)
    os.makedirs(velocities_path, exist_ok=True)
    os.makedirs(costs_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)
    os.makedirs(time_steps_path, exist_ok=True)
    os.makedirs(os.path.join(opt_vars_path, 'lambda_var'), exist_ok=True)
    os.makedirs(os.path.join(opt_vars_path, 't_var'), exist_ok=True)
    
    return results_dir
