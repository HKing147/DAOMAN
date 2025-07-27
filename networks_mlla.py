import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mlla import SelfMultiHeadAttention


def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)



# Different agents have different observation dimensions and action dimensions, so we need to use 'agent_id' to distinguish them
class Actor(nn.Module):
    def __init__(self, args, agent_id, attn=None):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id], args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 2)  # move
        # self.fc4 = nn.Linear(args.hidden_dim, 3)  # offload
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            # orthogonal_init(self.fc4)
        # self.encoder = Encoder(
        #     args.obs_dim_n[agent_id], n_block=2, n_embd=128, n_head=4, n_agent=3)
        # self.use_attention = args.use_attention
        # self.attn = attn
        # if args.use_attention:  # 使用注意力模块
        #     self.attn = MultiHeadAttention(
        #         args.obs_dim_n[agent_id], len(args.obs_dim_n))

    def forward(self, x):
        # x = self.encoder(x)
        # if self.use_attention:  # 使用注意力模块
        #     x = F.relu(self.attn(x))
        # if self.attn is not None:
        #     x = F.relu(self.attn(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        a = self.max_action * torch.tanh(self.fc3(x))  # move
        return a
        # b = F.softmax(self.fc4(x), dim=1)  # offload
        # # print(f"{a.shape = }")
        # # print(f"{b.shape = }")
        # return torch.cat([a, b], dim=-1)


class Critic_MADDPG(nn.Module):
    def __init__(self, args):
        super(Critic_MADDPG, self).__init__()
        self.fc1 = nn.Linear(sum(args.obs_dim_n) +
                             sum(args.action_dim_n), args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s, a):
        s = torch.cat(s, dim=1)
        a = torch.cat(a, dim=1)
        s_a = torch.cat([s, a], dim=1)

        q = F.relu(self.fc1(s_a))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q


class Critic_MATD3(nn.Module):
    def __init__(self, args, attn=None):
        super(Critic_MATD3, self).__init__()
        # self.encoder = Encoder(
        #     sum(args.obs_dim_n) +
        #     sum(args.action_dim_n), n_block=2, n_embd=256, n_head=4, n_agent=3)
        n_embd = 128
        n_head = 4
        n_agent = 3
        # self.attn = None
        self.attn = SelfMultiHeadAttention(args.obs_dim_n[0] + args.action_dim_n[0],2*args.hidden_dim,num_heads=4,dropout=0)
        # self.attn = SelfMultiHeadAttention(args.obs_dim_n[0] + args.action_dim_n[0],n_embd,num_heads=4,dropout=0.2)
        # self.fc1 = nn.Linear(n_embd*n_agent, args.hidden_dim)
        self.fc1 = nn.Linear(sum(args.obs_dim_n) +
                             sum(args.action_dim_n), args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

        # self.fc4 = nn.Linear(n_embd*n_agent, args.hidden_dim)
        self.fc4 = nn.Linear(sum(args.obs_dim_n) +
                             sum(args.action_dim_n), args.hidden_dim)
        self.fc5 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc6 = nn.Linear(args.hidden_dim, 1)
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)
            orthogonal_init(self.fc5)
            orthogonal_init(self.fc6)
        # self.use_attention = args.use_attention
        # if args.use_attention:  # 使用注意力模块
        #     self.attn = MultiHeadAttention(sum(args.obs_dim_n) +
        #                                    sum(args.action_dim_n), len(args.obs_dim_n))

    def forward(self, s, a):
        # (n_agents, batch_size, dim) List
        # print(f"{s = }")
        # print(f"{a = }")
        # s = torch.cat(s, dim=1) # matd3
        # a = torch.cat(a, dim=1) # matd3
        # matd3_attn (batch_size, n_agents, s_dim)
        s = torch.stack(s).transpose(0, 1)
        # matd3_attn (batch_size, n_agents, a_dim)
        a = torch.stack(a).transpose(0, 1)
        # (batch_size, n_agents, obs_dim+action_dim)
        s_a = torch.cat([s, a], dim=-1)
        # print(f"{s_a.shape = }")
        # s_a = self.encoder(s_a)
        if self.attn is not None:  # 使用注意力模块
            # s_a = F.relu(self.attn(s_a))
            s_a = self.attn(s_a)

        # (batch_size, n_agents*(obs_dim+action_dim) )
        s_a = s_a.reshape(s_a.shape[0], -1)

        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(s_a))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, s, a):
        # s = torch.cat(s, dim=1)
        # a = torch.cat(a, dim=1)
        # matd3_attn (batch_size, n_agents, s_dim)
        s = torch.stack(s).transpose(0, 1)
        # matd3_attn (batch_size, n_agents, a_dim)
        a = torch.stack(a).transpose(0, 1)
        # (batch_size, n_agents, obs_dim+action_dim)
        s_a = torch.cat([s, a], dim=-1)

        if self.attn is not None:  # 使用注意力模块
            # s_a = F.relu(self.attn(s_a))
            s_a = self.attn(s_a)
        
        # (batch_size, n_agents*(obs_dim+action_dim) )
        s_a = s_a.reshape(s_a.shape[0], -1)

        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        return q1
