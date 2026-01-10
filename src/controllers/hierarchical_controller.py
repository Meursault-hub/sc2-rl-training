from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th
from .basic_controller import BasicMAC
import torch.nn.functional as F

class HierarchicalMAC(BasicMAC):
    def __init__(self, scheme, groups, args):
        self.goal_interval = getattr(args, "goal_interval", 10)
        self.goal_dim = getattr(args, "goal_dim", 5)
        
        super().__init__(scheme, groups, args)
        
        # 构建 Manager (Goal) Agent
        goal_input_shape = self._get_input_shape_core(scheme) 
        self.goal_agent = agent_REGISTRY["goal_rnn"](goal_input_shape, args)

        self.goal_hidden_states = None
        self.current_goals = None 
        self.current_logits = None # [新增] 用于存储 Logits 以计算熵

    # --- [关键修改] 注册数据格式，让 Buffer 知道要存 "goals" ---
    def get_extra_scheme_shape(self, scheme):
        return {
            "goals": {"vshape": (self.goal_dim,), "dtype": th.float32},
        }

    def _get_input_shape(self, scheme):
        # [修改] Worker 输入不再直接拼接 Goal，由 Hypernetwork 内部处理
        # 所以这里只返回 core shape
        return self._get_input_shape_core(scheme)

    def _get_input_shape_core(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

    def init_hidden(self, batch_size):
        super().init_hidden(batch_size)
        # 确保初始化在正确的设备上
        self.goal_hidden_states = self.goal_agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1).to(self.args.device)
        self.current_goals = th.zeros(batch_size, self.n_agents, self.goal_dim).to(self.args.device)

    def forward(self, ep_batch, t, test_mode=False):
        bs = ep_batch.batch_size
        base_inputs = self._build_inputs_base(ep_batch, t) 
        
        if base_inputs.device != self.args.device:
            base_inputs = base_inputs.to(self.args.device)

        # 1. 更新 Goal (Manager)
        if t % self.goal_interval == 0:
            logits, self.goal_hidden_states = self.goal_agent(base_inputs, self.goal_hidden_states)
            self.current_logits = logits # 保存 Logits

            # [修改] 使用 Gumbel-Softmax 生成离散 Goal
            if test_mode:
                # 测试时：Hard Argmax (One-hot)
                goal_idx = logits.argmax(dim=-1, keepdim=True)
                goals = F.one_hot(goal_idx, num_classes=self.goal_dim).float()
            else:
                # 训练时：Gumbel-Softmax (Hard=True 表示输出 One-hot，但梯度可导)
                goals = F.gumbel_softmax(logits, tau=1.0, hard=True)

        # --- [关键修改] 将 Goal 放入输出字典，Runner 会自动收集存储 ---
        self.mac_output_extra = {
            "goals": self.current_goals
        }

        # 2. 运行 Worker
        goals_reshaped = self.current_goals.view(bs * self.n_agents, -1)
        
        avail_actions = ep_batch["avail_actions"][:, t]
        # [修改] 将 goal 作为参数传入，而不是拼接在 inputs 里
        agent_outs, self.hidden_states = self.agent(base_inputs, self.hidden_states, goal=goals_reshaped)

        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def _build_inputs_base(self, batch, t):
        # 复制自 basic_controller，构建不含 goal 的基础输入
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
        self.agent.cuda()
        self.goal_agent.cuda()