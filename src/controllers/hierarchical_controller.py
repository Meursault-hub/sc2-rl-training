from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from .basic_controller import BasicMAC # 继承基础控制器

class HierarchicalMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        # 1. 设置分层参数
        self.goal_interval = getattr(args, "goal_interval", 10) # 目标更新间隔，默认10步
        self.goal_dim = getattr(args, "goal_dim", 5)            # 目标向量维度
        
        # 2. 初始化父类 (这会构建 Worker Agent)
        super().__init__(scheme, groups, args)
        
        # 3. 构建 Manager (Goal) Agent
        # 注意：这里假设 obs 对于 High/Low level 是一样的
        goal_input_shape = self._get_input_shape_core(scheme) 
        self.goal_agent = agent_REGISTRY["goal_rnn"](goal_input_shape, args) # 需要在 agent registry 注册 goal_rnn

        # 状态存储
        self.goal_hidden_states = None
        self.current_goals = None # 存储当前所有 Agent 的目标

    # --- 重写: 计算 Worker Agent 的输入维度 ---
    def _get_input_shape(self, scheme):
        # Worker 的输入 = 原始输入 + Goal 向量
        input_shape = self._get_input_shape_core(scheme)
        input_shape += self.goal_dim 
        return input_shape

    # 辅助函数：获取原始观测维度 (不含 Goal)
    def _get_input_shape_core(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

    # --- 重写: 状态初始化 ---
    def init_hidden(self, batch_size):
        # 初始化 Worker 的隐藏状态
        super().init_hidden(batch_size)
        # 初始化 Goal Agent 的隐藏状态
        self.goal_hidden_states = self.goal_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1).to(self.args.device)
        # 初始化目标为 0
        self.current_goals = th.zeros(batch_size, self.n_agents, self.goal_dim).to(self.args.device)

    # --- 核心修改: Forward 逻辑 ---
    def forward(self, ep_batch, t, test_mode=False):
        bs = ep_batch.batch_size
        
        # 1. 构建基础输入 (Obs + Last Action + ID)
        # 注意：这里调用的是 BasicMAC 里的 _build_inputs，它只构建基础部分
        # 因为我们不想在 _build_inputs 里拼接旧的 Goal，我们要在这里控制 Goal 的更新
        base_inputs = self._build_inputs_base(ep_batch, t) 

        # 2. High-Level Strategy: 更新目标 (每隔 goal_interval 步 或 第一步)
        if t % self.goal_interval == 0:
            # 使用 base_inputs 作为 Goal Agent 的输入
            goals, self.goal_hidden_states = self.goal_agent(base_inputs, self.goal_hidden_states)
            self.current_goals = goals  # Detach 这里的梯度传递通常根据具体算法决定，简单版截断梯度
            
            # 如果是 HRL 训练，可能需要返回 goal 用于计算上层奖励，这里仅展示前向传播

        # 3. 拼接目标: 构造 Worker Agent 的完整输入
        # self.current_goals shape: [batch, n_agents, goal_dim]
        # base_inputs shape: [batch * n_agents, input_dim]
        
        goals_reshaped = self.current_goals.view(bs * self.n_agents, -1)
        worker_inputs = th.cat([base_inputs, goals_reshaped], dim=1)

        # 4. Low-Level Execution: 运行 Worker Agent (RNNAgent_New)
        avail_actions = ep_batch["avail_actions"][:, t]
        
        # 调用你的 RNNAgent_New
        agent_outs, self.hidden_states = self.agent(worker_inputs, self.hidden_states)

        # --- 以下逻辑与 BasicMAC 相同 (Softmax / Masking) ---
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    # 复制一份 BasicMAC 的 _build_inputs 但改名为 _build_inputs_base 以免冲突
    def _build_inputs_base(self, batch, t):
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t]) 
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs
    

    def cuda(self):
        self.agent.cuda()      # 搬运 Worker Agent (父类逻辑)
        self.goal_agent.cuda() # [新增] 搬运 Goal Agent