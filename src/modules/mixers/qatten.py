# From https://github.com/wjh720/QPLEX/, added here for convenience.
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle as pkl


class QattenMixer(nn.Module):
    def __init__(self, args):
        super(QattenMixer, self).__init__()

        self.name = 'qatten'
        self.args = args
        self.n_agents = args.n_agents                       # 智能体数量
        self.state_dim = int(np.prod(args.state_shape))     # 全局状态维度
        self.unit_dim = args.unit_dim                       # 单个智能体的局部特征维度
        self.n_actions = args.n_actions                     # 单个智能体的动作数
        self.sa_dim = self.state_dim + self.n_agents * self.n_actions
        self.n_head = args.n_head  # attention head num

        self.embed_dim = args.mixing_embed_dim              # 注意力嵌入维度
        self.attend_reg_coef = args.attend_reg_coef         # 注意力logits的正则化系数

        self.key_extractors = nn.ModuleList()               # 每个头的Key提取器（智能体局部特征→嵌入）
        self.selector_extractors = nn.ModuleList()          # 每个头的Query提取器（全局状态→嵌入）

        # 分超网络层数（1层/2层）构建Query提取器
        if getattr(args, "hypernet_layers", 1) == 1:
            for i in range(self.n_head):  # 为每个注意力头创建提取器
                self.selector_extractors.append(nn.Linear(self.state_dim, self.embed_dim, bias=False))  # query
                if self.args.nonlinear:   # 非线性模式：拼接智能体自身Q值到局部特征
                    self.key_extractors.append(nn.Linear(self.unit_dim + 1, self.embed_dim, bias=False))  # key
                else:                     # 线性模式：仅用局部特征
                    self.key_extractors.append(nn.Linear(self.unit_dim, self.embed_dim, bias=False))  # key
            if self.args.weighted_head:   # 若开启多头加权，构建“头权重生成网络”（全局状态→各头权重）
                self.hyper_w_head = nn.Linear(self.state_dim, self.n_head)
                # self.hyper_w_head = nn.Linear(self.state_dim, self.embed_dim * self.n_head)
        elif getattr(args, "hypernet_layers", 1) == 2:  # 2层超网络：Query提取器增加非线性（Linear+ReLU+Linear），Key仍为1层
            hypernet_embed = self.args.hypernet_embed
            for i in range(self.n_head):  # multi-head attention
                selector_nn = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                            nn.ReLU(),
                                            nn.Linear(hypernet_embed, self.embed_dim, bias=False))
                self.selector_extractors.append(selector_nn)  # query
                if self.args.nonlinear:  # add qs
                    self.key_extractors.append(nn.Linear(self.unit_dim + 1, self.embed_dim, bias=False))  # key
                else:
                    self.key_extractors.append(nn.Linear(self.unit_dim, self.embed_dim, bias=False))  # key
            if self.args.weighted_head:   # 2层的头权重生成网络
                self.hyper_w_head = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                                  nn.ReLU(),
                                                  nn.Linear(hypernet_embed, self.n_head))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 embednet layers is not implemented!")
        else:
            raise Exception("Error setting number of embednet layers.")

        if self.args.state_bias:  # 偏置也采用网络模拟，非固定常数
            # V(s) instead of a bias for the last layers
            self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states, actions):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        unit_states = states[:, : self.unit_dim * self.n_agents]  # get agent own features from state
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)
        unit_states = unit_states.permute(1, 0, 2)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)  # agent_qs: (batch_size, 1, agent_num)

        if self.args.nonlinear:
            unit_states = th.cat((unit_states, agent_qs.permute(2, 0, 1)), dim=2)
        # states: (batch_size, state_dim)
        all_head_selectors = [sel_ext(states) for sel_ext in self.selector_extractors]
        # all_head_selectors: (head_num, batch_size, embed_dim)
        # unit_states: (agent_num, batch_size, unit_dim)
        all_head_keys = [[k_ext(enc) for enc in unit_states] for k_ext in self.key_extractors]
        # all_head_keys: (head_num, agent_num, batch_size, embed_dim)

        # calculate attention per head
        head_qs = []
        head_attend_logits = []
        head_attend_weights = []
        for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
            # curr_head_keys: (agent_num, batch_size, embed_dim)
            # curr_head_selector: (batch_size, embed_dim)

            # (batch_size, 1, embed_dim) * (batch_size, embed_dim, agent_num)
            attend_logits = th.matmul(curr_head_selector.view(-1, 1, self.embed_dim),
                                      th.stack(curr_head_keys).permute(1, 2, 0))
            # attend_logits: (batch_size, 1, agent_num)
            # scale dot-products by size of key (from Attention is All You Need)
            scaled_attend_logits = attend_logits / np.sqrt(self.embed_dim)
            if self.args.mask_dead:
                # actions: (episode_batch, episode_length - 1, agent_num, 1)
                actions = actions.reshape(-1, 1, self.n_agents)
                # actions: (batch_size, 1, agent_num)
                scaled_attend_logits[actions == 0] = -99999999  # action == 0 means the unit is dead
            attend_weights = F.softmax(scaled_attend_logits, dim=2)  # (batch_size, 1, agent_num)
            # (batch_size, 1, agent_num) * (batch_size, 1, agent_num)
            head_q = (agent_qs * attend_weights).sum(dim=2)
            head_qs.append(head_q)
            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)
        if self.args.state_bias:
            # State-dependent bias
            v = self.V(states).view(-1, 1)  # v: (bs, 1)
            # head_qs: [head_num, bs, 1]
            if self.args.weighted_head:
                w_head = th.abs(self.hyper_w_head(states))  # w_head: (bs, head_num)
                w_head = w_head.view(-1, self.n_head, 1)  # w_head: (bs, head_num, 1)
                y = th.stack(head_qs).permute(1, 0, 2)  # head_qs: (head_num, bs, 1); y: (bs, head_num, 1)
                y = (w_head * y).sum(dim=1) + v  # y: (bs, 1)
            else:
                y = th.stack(head_qs).sum(dim=0) + v  # y: (bs, 1)
        else:
            if self.args.weighted_head:
                w_head = th.abs(self.hyper_w_head(states))  # w_head: (bs, head_num)
                w_head = w_head.view(-1, self.n_head, 1)  # w_head: (bs, head_num, 1)
                y = th.stack(head_qs).permute(1, 0, 2)  # head_qs: (head_num, bs, 1); y: (bs, head_num, 1)
                y = (w_head * y).sum(dim=1)  # y: (bs, 1)
            else:
                y = th.stack(head_qs).sum(dim=0)  # y: (bs, 1)
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        # regularize magnitude of attention logits
        attend_mag_regs = self.attend_reg_coef * sum((logit ** 2).mean() for logit in head_attend_logits)
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze(dim=1).sum(1).mean()) for probs in head_attend_weights]
        return q_tot, attend_mag_regs, head_entropies
