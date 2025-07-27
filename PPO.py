import argparse
from collections import namedtuple

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    '--gamma', type=float, default=0.9, metavar='G', help='discount factor (default: 0.9)')
parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
parser.add_argument('--render', action='store_true', default=False, help='render the environment')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    metavar='N',
    help='interval between training status logs (default: 10)')
args = parser.parse_args()

# env = gym.make('Pendulum-v0').unwrapped
# num_state = env.observation_space.shape[0]
# num_action = env.action_space.shape[0]
# torch.manual_seed(args.seed)
# env.seed(args.seed)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state'])
TrainRecord = namedtuple('TrainRecord', ['episode', 'reward'])


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.softmax(self.out(x), dim = -1)
        return x


class Critic(nn.Module):
    def __init__(self, state_dim,  hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.state_value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 10
    buffer_capacity, batch_size = 1000, 256

    def __init__(self, state_dim, action_dim, hidden_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init, device, multi_buffer= False):
        super(PPO, self).__init__()
        self.device = device
        self.actor_net = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic_net = Critic(state_dim, hidden_dim).to(device)
        if multi_buffer:
            self.buffer = [[] for _ in range(3)]
            self.counter = [0 for _ in range(3)]
            self.training_step = [0 for _ in range(3)]
        else:
            self.buffer = []
            self.counter = 0
            self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=lr_actor)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=lr_critic)
        # if not os.path.exists('../param'):
        #     os.makedirs('../param/net_param')
        #     os.makedirs('../param/img')

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.actor_net(state)
        dist = Categorical(prob)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        # action = action.clamp(-2, 2)
        return action.item(), action_log_prob.item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    # def save_param(self):
    #     torch.save(self.anet.state_dict(), 'param/ppo_anet_params.pkl')
    #     torch.save(self.cnet.state_dict(), 'param/ppo_cnet_params.pkl')

    def store_transition(self, transition, idx = -1):
        if idx == -1:
            self.buffer.append(transition)
            self.counter += 1
            return self.counter % self.buffer_capacity == 0
        else:
            self.buffer[idx].append(transition)
            self.counter[idx] += 1
            return self.counter[idx] % self.buffer_capacity == 0

    def update(self, idx = -1):
        if idx == -1:
            self.training_step += 1
            buffer = self.buffer
        else:
            self.training_step[idx] += 1
            buffer = self.buffer[idx]

        state = torch.tensor([t.state for t in buffer], dtype=torch.float).to(self.device)
        action = torch.tensor([t.action for t in buffer], dtype=torch.float).view(-1, 1).to(self.device)
        reward = torch.tensor([t.reward for t in buffer], dtype=torch.float).view(-1, 1).to(self.device)
        next_state = torch.tensor([t.next_state for t in buffer], dtype=torch.float).to(self.device)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in buffer], dtype=torch.float).view(-1, 1).to(self.device)
        reward = (reward - reward.mean()) / (reward.std() + 1e-5)
        with torch.no_grad():
            target_v = reward + args.gamma * self.critic_net(next_state)

        advantage = (target_v - self.critic_net(state)).detach()
        for _ in range(self.ppo_epoch):  # iteration ppo_epoch
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                # epoch iteration, PPO core!!!
                prob = self.actor_net(state[index])
                n = Categorical(prob)
                action_log_prob = n.log_prob(action[index])
                ratio = torch.exp(action_log_prob - old_action_log_prob[index])

                L1 = ratio * advantage[index]
                L2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage[index]
                action_loss = -torch.min(L1, L2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                value_loss = F.smooth_l1_loss(self.critic_net(state[index]), target_v[index])
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()

        del buffer[:]


def main():
    agent = PPO()

    training_records = []
    running_reward = -1000

    for i_epoch in range(1000):
        score = 0
        state = env.reset()
        if args.render: env.render()
        for t in range(200):
            action, action_log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step([action])
            trans = Transition(state, action, (reward + 8) / 8, action_log_prob, next_state)
            if args.render: env.render()
            if agent.store_transition(trans):
                agent.update()
            score += reward
            state = next_state

        running_reward = running_reward * 0.9 + score * 0.1
        training_records.append(TrainRecord(i_epoch, running_reward))
        if i_epoch % 10 == 0:
            print("Epoch {}, Moving average score is: {:.2f} ".format(i_epoch, running_reward))
        if running_reward > -200:
            print("Solved! Moving average score is now {}!".format(running_reward))
            env.close()
            # agent.save_param()
            break


if __name__ == '__main__':
    main()