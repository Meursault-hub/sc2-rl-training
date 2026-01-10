import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent_New(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent_New, self).__init__()
        self.args = args

        # --- Hypernetwork 配置 ---
        self.goal_dim = args.goal_dim
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.hyper_embed_dim = getattr(args, "hypernet_embed", 64)

        # [新增 1] 输入层归一化 (Input Normalization)
        # SMAC 的输入包含未归一化的数值(如血量)，这会导致 Hypernet 生成的权重效果被放大
        self.input_norm = nn.LayerNorm(input_shape)

        # 1. 定义 Hypernetwork
        self.n_weights = input_shape * self.rnn_hidden_dim
        self.n_bias = self.rnn_hidden_dim

        self.hyper_net = nn.Sequential(
            nn.Linear(self.goal_dim, self.hyper_embed_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_embed_dim, self.n_weights + self.n_bias)
        )

        # 2. Worker 的后续网络
        # 在动态层之后再次归一化
        self.layer_norm_2 = nn.LayerNorm(args.rnn_hidden_dim)

        self.feat_extract_2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        )

        # GRU 核心
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # 3. Dueling Heads
        self.val_head = nn.Linear(args.rnn_hidden_dim, 1)
        self.adv_head = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.hyper_net[0].weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, goal):
        # inputs: [batch * n_agents, input_shape]
        bs, input_dim = inputs.shape

        # [应用 1] 先对原始输入做归一化，防止大数值(如HP)引起爆炸
        inputs = self.input_norm(inputs)

        # --- Hypernetwork 生成参数 ---
        hyper_out = self.hyper_net(goal)
        
        # [关键应用 2] 硬限制 (Hard Constraint)
        # 强制将生成的参数限制在 [-3.0, 3.0] 区间内
        # 无论 Hypernet 的梯度如何爆炸，生成的 Worker 权重永远是安全的
        hyper_out = F.tanh(hyper_out) * 3.0

        weights = hyper_out[:, :self.n_weights].view(bs, input_dim, self.rnn_hidden_dim)
        bias = hyper_out[:, self.n_weights:].view(bs, self.rnn_hidden_dim)

        # --- 动态层计算 ---
        # x = inputs * weights + bias
        x = th.bmm(inputs.unsqueeze(1), weights).squeeze(1) + bias
        
        # [应用 3] 层归一化
        x = self.layer_norm_2(x)

        # --- 后续计算 ---
        x = self.feat_extract_2(x)
        
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        
        val = self.val_head(h)
        adv = self.adv_head(h)
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        
        return q, h