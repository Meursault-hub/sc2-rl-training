import torch.nn as nn
import torch.nn.functional as F

class GoalAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GoalAgent, self).__init__()
        self.args = args
        self.goal_dim = args.goal_dim # 需要在 config 中定义，例如 5 或 10

        # 特征提取
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        
        # 输出目标向量
        # 使用 tanh 将目标限制在 [-1, 1] 区间，保证数值稳定性
        self.goal_head = nn.Linear(args.rnn_hidden_dim, self.goal_dim)

    def init_hidden(self):
        # 初始化隐藏状态
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        
        # 生成目标
        goal = F.tanh(self.goal_head(h))
        return goal, h