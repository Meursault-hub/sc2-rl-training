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
        self.n_head = getattr(args, "n_head", 8) 

        self.embed_dim = args.mixing_embed_dim
        self.attend_reg_coef = args.attend_reg_coef

        # [安全锁 1] 输入归一化
        self.state_norm = nn.LayerNorm(self.state_dim)

        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()

        hypernet_embed = self.args.hypernet_embed
        for i in range(self.n_head):
            selector_nn = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim, bias=False)
            )
            self.selector_extractors.append(selector_nn)

            if self.args.nonlinear:
                input_dim = self.unit_dim + 1
            else:
                input_dim = self.unit_dim
            
            key_hidden_dim = self.embed_dim * 2 
            key_net = nn.Sequential(
                nn.Linear(input_dim, key_hidden_dim),
                nn.ReLU(),
                nn.Linear(key_hidden_dim, self.embed_dim, bias=False)
            )
            self.key_extractors.append(key_net)

        # [安全锁 2] 加固 Head Weight 网络
        if self.args.weighted_head:
            self.hyper_w_head = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.LayerNorm(hypernet_embed), 
                nn.Linear(hypernet_embed, self.n_head)
            )

        # [安全锁 3] V(s) 网络结构强化
        if self.args.state_bias:
            v_hidden_dim = self.embed_dim * 4 
            self.V_net_1 = nn.Linear(self.state_dim, v_hidden_dim)
            self.V_ln = nn.LayerNorm(v_hidden_dim) 
            self.V_net_2 = nn.Linear(v_hidden_dim, 1)

        # [安全锁 4] 强制极小初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, agent_qs, states, actions):
        bs = agent_qs.size(0)
        
        # 1. State 归一化
        states = states.reshape(-1, self.state_dim)
        states = self.state_norm(states)

        unit_states = states[:, : self.unit_dim * self.n_agents]
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)
        unit_states = unit_states.permute(1, 0, 2)

        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        if self.args.nonlinear:
            unit_states = th.cat((unit_states, agent_qs.permute(2, 0, 1)), dim=2)

        all_head_selectors = [sel_ext(states) for sel_ext in self.selector_extractors]
        all_head_keys = [[k_ext(enc) for enc in unit_states] for k_ext in self.key_extractors]

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

        if self.args.state_bias:
            v_hidden = self.V_net_1(states)
            v_hidden = F.relu(self.V_ln(v_hidden))
            v = self.V_net_2(v_hidden).view(-1, 1)
            
            # [关键修正] V(s) 锁死
            v = th.clamp(v, min=-200.0, max=200.0)

            if self.args.weighted_head:
                w_head = th.abs(self.hyper_w_head(states))
                w_head = w_head.view(-1, self.n_head, 1)
                # [关键修正] Weight 锁死
                w_head = th.clamp(w_head, max=10.0) 

                y = th.stack(head_qs).permute(1, 0, 2)
                y = (w_head * y).sum(dim=1) + v
            else:
                y = th.stack(head_qs).sum(dim=0) + v
        else:
            y = th.stack(head_qs).sum(dim=0)

        q_tot = y.view(bs, -1, 1)

        # ==========================================================
        # [终极核武] Q_tot 最终锁死 (The Ultimate Clamp)
        # 无论前面发生了什么，这里强制把 Q_tot 限制在 [-500, 500]
        # 星际争霸最大回报通常不超过 100，所以这个范围绝对安全且足够
        # ==========================================================
        q_tot = th.clamp(q_tot, min=-500.0, max=500.0)
        
        # --- Debug 打印 (如果还炸，请取消注释查看是谁在变大) ---
        # if th.abs(q_tot).max() > 400:
        #     print(f"WARNING: Large Q_tot detected! Max: {q_tot.max().item()}")
        #     print(f"V(s) stats: Min {v.min().item()} Max {v.max().item()}")
        #     if self.args.weighted_head:
        #         print(f"W_head stats: Max {w_head.max().item()}")
        # --------------------------------------------------------

        attend_mag_regs = self.attend_reg_coef * sum((logit ** 2).mean() for logit in head_attend_logits)
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze(dim=1).sum(1).mean()) for probs in head_attend_weights]
        
        return q_tot, attend_mag_regs, head_entropies