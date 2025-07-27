import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
from networks_mlla import Actor, Critic_MATD3


class MATD3(nn.Module):
    def __init__(self, args, agent_id, attn=None):
        super(MATD3, self).__init__()
        self.device = args.device
        self.attn = attn
        self.N = args.N
        self.agent_id = agent_id
        self.max_action = args.max_action
        self.action_dim = args.action_dim_n[agent_id]
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.tau = args.tau
        self.use_grad_clip = args.use_grad_clip
        self.policy_noise = args.policy_noise
        self.noise_clip = args.noise_clip
        self.policy_update_freq = args.policy_update_freq
        self.actor_pointer = 0
        # Create an individual actor and critic for each agent according to the 'agent_id'
        self.actor = Actor(args, agent_id)
        self.critic = Critic_MATD3(args, self.attn)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr_a)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr_c)
        self.critic_metrics = []
        self.actor_metrics = []


    def choose_action(self, obs, noise_std, mask=None):
        obs = torch.unsqueeze(torch.tensor(
            obs, dtype=torch.float), 0).to(self.device)
        # print(f"{obs.shape = }")
        a = self.actor(obs).data.cpu().numpy().flatten()
        a[:2] = (a[:2] + np.random.normal(0, noise_std, size=2)
                 ).clip(-self.max_action, self.max_action)  # clip move_action
        # a[2:] = (a[2:] + np.random.normal(0.01, 0.2, size=3)
        #          ).clip(0, 1)  # clip offload_action
        # if mask is not None:
        #     a[2:][np.where(np.array(mask) == 0)] = 0
        #     if sum(a[2:]) == 0:  # 防止mask之后全为0
        #         a[2:][np.where(np.array(mask) == 1)] = 1
        return a

    def train(self, replay_buffer, agent_n):
        self.actor_pointer += 1
        batch_obs_n, batch_a_n, batch_r_n, batch_obs_next_n, batch_done_n = replay_buffer.sample()

        # Compute target_Q
        with torch.no_grad():  # target_Q has no gradient
            batch_a_next_n = []
            for i in range(self.N):
                batch_a_next = agent_n[i].actor_target(batch_obs_next_n[i])
                noise = (torch.randn_like(batch_a_next) *
                         self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                batch_a_next = (
                    batch_a_next + noise).clamp(-self.max_action, self.max_action)
                batch_a_next_n.append(batch_a_next)

            # print(f"{len(batch_obs_next_n) = } {batch_obs_next_n[0].shape = }")
            # print(f"{len(batch_a_next_n) = } {batch_a_next_n[0].shape = }")
            Q1_next, Q2_next = self.critic_target(
                batch_obs_next_n, batch_a_next_n)
            target_Q = batch_r_n[self.agent_id] + self.gamma * (
                1 - batch_done_n[self.agent_id]) * torch.min(Q1_next, Q2_next)  # shape:(batch_size,1)

        # Compute current_Q
        current_Q1, current_Q2 = self.critic(
            batch_obs_n, batch_a_n)  # shape:(batch_size,1)
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
            F.mse_loss(current_Q2, target_Q)
        self.critic_metrics.append(critic_loss.detach())
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_optimizer.step()

        if self.actor_pointer % self.policy_update_freq == 0:
            # Reselect the actions of the agent corresponding to 'agent_id', the actions of other agents remain unchanged
            batch_a_n[self.agent_id] = self.actor(batch_obs_n[self.agent_id])
            # Only use Q1
            actor_loss = - \
                self.critic.Q1(
                    batch_obs_n, batch_a_n).mean()
            self.actor_metrics.append(actor_loss.detach())
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_grad_clip:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_optimizer.step()

            # Softly update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_model(self, model_dir, episode, agent_id):
        # 需要保存的模型参数
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "epoch": episode
        }
        path_checkpoint = "{}/checkpoint_epoch_{}_agent_{}.pkl".format(
            model_dir, episode, agent_id)
        torch.save(checkpoint, path_checkpoint)

    def load_model(self, model_dir, episode, agent_id):
        path_checkpoint = "{}/checkpoint_epoch_{}_agent_{}.pkl".format(
            model_dir, episode, agent_id)
        checkpoint = torch.load(path_checkpoint)  # load文件
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer_state_dict"])
        # target 网络
        self.actor_target.load_state_dict(checkpoint["actor_state_dict"])
        self.critic_target.load_state_dict(checkpoint["critic_state_dict"])
        print(f"load {path_checkpoint} success...")
