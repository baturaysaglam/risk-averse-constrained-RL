import torch
import torch.nn as nn
import torch.optim as optim


# Implementation of the Proximal Policy Optimization (PPO) algorithm
# Paper: https://arxiv.org/abs/1707.06347
# Note: This implementation is based on the repository: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

class PPO:
    def __init__(self,
                 policy,
                 clip_param,
                 n_epochs,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 adam_eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):
        self.policy = policy

        self.clip_param = clip_param
        self.n_epochs = n_epochs
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=adam_eps)

    def update_parameters(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for _ in range(self.n_epochs):
            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, actions_batch, _, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, _ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy = self.policy.evaluate_actions(obs_batch, masks_batch, actions_batch)

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

                # Compute the clipped surrogate (policy) loss
                surr_1 = ratio * adv_targ
                surr_2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

                action_loss = -torch.min(surr_1, surr_2).mean()

                # Compute the value loss
                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                # Optimize the policy
                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def save_checkpoint(self, file_name, obs_rms, opt_vars=None):
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'obs_rms': obs_rms
        }

        # Add opt_vars if not None
        if opt_vars is not None:
            checkpoint['lambda_var'] = opt_vars['lambda_var']
            checkpoint['t_var'] = opt_vars['t_var']

        torch.save(checkpoint, file_name)

    def load_checkpoint(self, file_name):
        checkpoint = torch.load(file_name)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load opt_vars if they exist in the checkpoint
        lambda_var = checkpoint.get('lambda_var', None)
        t_var = checkpoint.get('t_var', None)
        opt_vars = {'lambda_var': lambda_var, 't_var': t_var} if lambda_var is not None and t_var is not None else None

        return checkpoint['obs_rms'], opt_vars
