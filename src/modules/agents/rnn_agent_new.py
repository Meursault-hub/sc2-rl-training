import torch as th
import torch.nn as nn
import torch.nn.functional as F

class RNNAgent_New(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent_New, self).__init__()
        self.args = args

        # --- [关键修改 1] 增强型特征提取器 ---
        # 原版: Linear(input -> hidden)
        # 新版: Linear -> ReLU -> Linear (更强的局部感知)
        self.feat_extract = nn.Sequential(
            nn.Linear(input_shape, args.rnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        )

        # GRU 保持不变，这是处理序列的核心
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)

        # --- [关键修改 2] Dueling Network (决斗网络结构) ---
        # 将输出拆分为两部分：
        # 1. Value Head: 评估当前状态好不好 (输出维度 1)
        # 2. Advantage Head: 评估每个动作好不好 (输出维度 n_actions)
        
        self.val_head = nn.Linear(args.rnn_hidden_dim, 1)
        self.adv_head = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.feat_extract[0].weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # 1. 特征提取
        x = F.relu(self.feat_extract(inputs))
        
        # 2. RNN 记忆处理
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        
        # 3. Dueling 计算
        # V(s): 状态价值
        val = self.val_head(h)
        
        # A(s, a): 动作优势
        adv = self.adv_head(h)
        
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        # 这种聚合方式能提高稳定性
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        
        return q, h