import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QattenMixer_New(nn.Module):
    def __init__(self, args):
        super(QattenMixer_New, self).__init__()

        self.name = 'qatten_new'
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.unit_dim = args.unit_dim
        self.n_actions = args.n_actions
        self.sa_dim = self.state_dim + self.n_agents * self.n_actions
        self.n_head = getattr(args, "n_head", 8) # 默认4头，建议在config里改为8头

        self.embed_dim = args.mixing_embed_dim
        self.attend_reg_coef = args.attend_reg_coef

        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()

        # --- Query (Selector) 提取器 ---
        hypernet_embed = self.args.hypernet_embed
        for i in range(self.n_head):
            # 保持原有的双层结构
            selector_nn = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                        nn.ReLU(),
                                        nn.Linear(hypernet_embed, self.embed_dim, bias=False))
            self.selector_extractors.append(selector_nn)

            # --- [关键修改 1] Deep Key Extractor ---
            # 原版只是一个 nn.Linear，能力太弱。
            # 这里改为 MLP，赋予 Key 提取非线性能力。
            if self.args.nonlinear:
                input_dim = self.unit_dim + 1
            else:
                input_dim = self.unit_dim
            
            # 中间层维度稍微放大，保证特征不丢失
            key_hidden_dim = self.embed_dim * 2 
            
            key_net = nn.Sequential(
                nn.Linear(input_dim, key_hidden_dim),
                nn.ReLU(),
                nn.Linear(key_hidden_dim, self.embed_dim, bias=False)
            )
            self.key_extractors.append(key_net)

        # 头权重网络
        if self.args.weighted_head:
            self.hyper_w_head = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                              nn.ReLU(),
                                              nn.Linear(hypernet_embed, self.n_head))

        # --- [关键修改 2] Wide V-Network ---
        # 扩大全局偏差网络 V(s) 的容量
        if self.args.state_bias:
            v_hidden_dim = self.embed_dim * 4 # 扩大4倍
            self.V = nn.Sequential(nn.Linear(self.state_dim, v_hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(v_hidden_dim, 1))

    def forward(self, agent_qs, states, actions):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        unit_states = states[:, : self.unit_dim * self.n_agents]
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)
        unit_states = unit_states.permute(1, 0, 2)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        if self.args.nonlinear:
            unit_states = th.cat((unit_states, agent_qs.permute(2, 0, 1)), dim=2)

        # 提取 Query (Selector)
        all_head_selectors = [sel_ext(states) for sel_ext in self.selector_extractors]

        # 提取 Key (现在通过更强的 MLP 提取)
        all_head_keys = [[k_ext(enc) for enc in unit_states] for k_ext in self.key_extractors]

        # 计算注意力
        head_qs = []
        head_attend_logits = []
        head_attend_weights = []
        for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
            attend_logits = th.matmul(curr_head_selector.view(-1, 1, self.embed_dim),
                                      th.stack(curr_head_keys).permute(1, 2, 0))
            scaled_attend_logits = attend_logits / np.sqrt(self.embed_dim)
            if self.args.mask_dead:
                actions = actions.reshape(-1, 1, self.n_agents)
                scaled_attend_logits[actions == 0] = -99999999
            attend_weights = F.softmax(scaled_attend_logits, dim=2)
            head_q = (agent_qs * attend_weights).sum(dim=2)
            head_qs.append(head_q)
            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)

        # 计算 V(s) 并聚合
        if self.args.state_bias:
            v = self.V(states).view(-1, 1)
            if self.args.weighted_head:
                w_head = th.abs(self.hyper_w_head(states))
                w_head = w_head.view(-1, self.n_head, 1)
                y = th.stack(head_qs).permute(1, 0, 2)
                y = (w_head * y).sum(dim=1) + v
            else:
                y = th.stack(head_qs).sum(dim=0) + v
        else:
             # Fallback (通常都会开 state_bias)
            y = th.stack(head_qs).sum(dim=0)

        q_tot = y.view(bs, -1, 1)
        attend_mag_regs = self.attend_reg_coef * sum((logit ** 2).mean() for logit in head_attend_logits)
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze(dim=1).sum(1).mean()) for probs in head_attend_weights]
        
        return q_tot, attend_mag_regs, head_entropies