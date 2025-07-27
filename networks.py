import torch
import torch.nn as nn
import torch.nn.functional as F
from MultiHeadAttention import MultiHeadAttention
import math


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


'''
class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        # print(f'{query.shape = }')
        L, D = query.size()
        print(f"{L = } {D = }")
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(L, self.n_head, D //
                               self.n_head).transpose(0, 1)  # (B, nh, L, hs)
        q = self.query(query).view(L, self.n_head, D //
                                   self.n_head).transpose(0, 1)  # (B, nh, L, hs)
        v = self.value(value).view(L, self.n_head, D //
                                   self.n_head).transpose(0, 1)  # (B, nh, L, hs)
        print(f"{k.shape = }")
        print(f"{q.shape = }")
        print(f"{v.shape = }")
        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        print(f"{att.shape = }")
        if self.masked:
            att = att.masked_fill(self.mask[:, :L] == 0, float('-inf'))
        print(f"{att.shape = }")
        att = F.softmax(att, dim=-1)
        print(f"{att.shape = }")

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(0, 1).contiguous().view(L, D)

        # output projection
        y = self.proj(y)
        return y
'''


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()  # [batch, n_agents, obs_dim]

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D //
                               self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D //
                                   self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D //
                                   self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, L, D)

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(self,  obs_dim, n_block, n_embd, n_head, n_agent):
        super(Encoder, self).__init__()

        # self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        # self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
        #                                    init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(
            *[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])

        self.out_layer = nn.Sequential(nn.LayerNorm(n_embd),
                                       init_(nn.Linear(n_embd, obs_dim), activate=True), nn.GELU())

    def forward(self,  obs):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings

        rep = self.blocks(self.ln(x))

        rep = self.out_layer(rep)

        return rep


# Different agents have different observation dimensions and action dimensions, so we need to use 'agent_id' to distinguish them
class Actor(nn.Module):
    def __init__(self, args, agent_id, attn=None):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.obs_dim_n[agent_id], args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 2)  # move
        self.fc4 = nn.Linear(args.hidden_dim, 3)  # offload
        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)
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
        b = F.softmax(F.relu(self.fc4(x)), dim=1)  # offload
        # print(f"{a.shape = }")
        # print(f"{b.shape = }")
        return torch.cat([a, b], dim=-1)


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

        self.fc1 = nn.Linear(sum(args.obs_dim_n) +
                             sum(args.action_dim_n), args.hidden_dim)
        # self.fc1 = nn.Linear(args.obs_dim_n[0] +
        #                      args.action_dim_n[0], args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

        self.fc4 = nn.Linear(sum(args.obs_dim_n) +
                             sum(args.action_dim_n), args.hidden_dim)
        # self.fc4 = nn.Linear(args.obs_dim_n[0] +
        #                      args.action_dim_n[0], args.hidden_dim)
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
        self.attn = attn
        # if args.use_attention:  # 使用注意力模块
        #     self.attn = MultiHeadAttention(sum(args.obs_dim_n) +
        #                                    sum(args.action_dim_n), len(args.obs_dim_n))

    def forward(self, s, a):
        # (n_agents, batch_size, dim) List
        # print(f"{s = }")
        # print(f"{a = }")
        # s = torch.cat(s, dim=1)  # matd3
        # a = torch.cat(a, dim=1)  # matd3
        # s_a = torch.cat([s, a], dim=-1)
        #  (batch_size, n_agents, s_dim)
        s = torch.stack(s).transpose(0, 1)
        #  (batch_size, n_agents, a_dim)
        a = torch.stack(a).transpose(0, 1)
        s_a = torch.cat([s, a], dim=-1)
        s_a = s_a.reshape(s_a.shape[0], -1)
        # print(f"{s_a.shape = }")
        # s_a = self.encoder(s_a)
        if self.attn is not None:  # 使用注意力模块
            s_a = F.relu(self.attn(s_a))

        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.fc4(s_a))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2

    def Q1(self, s, a):
        # n_agent, batch, dim
        # s = torch.cat(s, dim=1)
        # a = torch.cat(a, dim=1)
        # s_a = torch.cat([s, a], dim=1)

        #  (batch_size, n_agents, s_dim)
        s = torch.stack(s).transpose(0, 1)
        #  (batch_size, n_agents, a_dim)
        a = torch.stack(a).transpose(0, 1)
        s_a = torch.cat([s, a], dim=-1)
        s_a = s_a.reshape(s_a.shape[0], -1)
        # s_a = self.encoder(s_a)

        if self.attn is not None:  # 使用注意力模块
            s_a = F.relu(self.attn(s_a))

        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        return q1
