import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.qatten_new import QattenMixer_New
import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F
import numpy as np

class FeudalLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.params = list(mac.parameters()) 

        # --- 1. 定义两个 Mixer ---
        # Worker Mixer: 用于评估内部奖励 (Intrinsic Reward)
        # Manager Mixer: 用于评估外部奖励 (Global Reward)
        if args.mixer == "qatten_new":
            self.mixer_worker = QattenMixer_New(args)
            self.mixer_manager = QattenMixer_New(args)
        else:
            raise ValueError("Feudal Learner strictly requires QAtten_New mixer")

        # Target Mixers
        self.target_mixer_worker = copy.deepcopy(self.mixer_worker)
        self.target_mixer_manager = copy.deepcopy(self.mixer_manager)

        # --- 2. 定义两个优化器 ---
        
        # Worker 参数: Agent + Worker Mixer
        self.worker_params = list(mac.agent.parameters()) + list(self.mixer_worker.parameters())
        self.worker_optimiser = RMSprop(params=self.worker_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # Manager 参数: Goal Agent + Manager Mixer
        self.manager_params = list(mac.goal_agent.parameters()) + list(self.mixer_manager.parameters())
        self.manager_optimiser = RMSprop(params=self.manager_params, lr=args.manager_lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # Target Networks (MAC)
        self.target_mac = copy.deepcopy(mac)
        self.last_target_update_episode = 0

        # --- 3. 辅助网络: 状态差分嵌入层 ---
        # 用于将 (s' - s) 映射到 Goal 维度，计算内部奖励
        obs_dim = scheme["obs"]["vshape"]
        self.state_embed = th.nn.Linear(obs_dim, args.goal_dim).to(args.device)
        # 这个嵌入层归 Worker 训练，因为它定义了Worker的"理解"
        self.worker_optimiser.add_param_group({'params': self.state_embed.parameters()})

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 通用数据
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        goals = batch["goals"][:, :-1]

        # ===================================================
        # 1. 前向传播与 Loss 计算 (只计算，不更新)
        # ===================================================
        
        # --- A. 计算 Worker Loss ---
        # 1. Intrinsic Reward
        with th.no_grad():
            obs_curr = batch["obs"][:, :-1]
            obs_next = batch["obs"][:, 1:]
            diff = obs_next - obs_curr
            diff_embed = self.state_embed(diff)
            detached_goals = goals.detach()
            intrinsic_rewards = F.cosine_similarity(diff_embed, detached_goals, dim=-1).unsqueeze(-1)
            intrinsic_rewards = intrinsic_rewards / 10.0

        # 2. Forward Worker
        mac_out_worker = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out_worker.append(agent_outs)
        mac_out_worker = th.stack(mac_out_worker, dim=1)

        # 3. Worker Mixer & Loss
        chosen_action_qvals_worker = th.gather(mac_out_worker[:, :-1], dim=3, index=actions).squeeze(3)
        chosen_action_qvals_worker, q_attend_regs_worker, _ = self.mixer_worker(chosen_action_qvals_worker, batch["state"][:, :-1], actions)

        with th.no_grad():
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            target_mac_out = th.stack(target_mac_out[1:], dim=1)
            target_mac_out[avail_actions[:, 1:] == 0] = -9999999
            
            target_max_qvals_worker = target_mac_out.max(dim=3)[0]
            target_next_actions = target_mac_out.max(dim=3, keepdim=True)[1]
            target_max_qvals_worker, _, _ = self.target_mixer_worker(target_max_qvals_worker, batch["state"][:, 1:], target_next_actions)

        targets_worker = intrinsic_rewards.sum(dim=2) + self.args.gamma * (1 - terminated) * target_max_qvals_worker
        td_error_worker = (chosen_action_qvals_worker - targets_worker.detach())
        loss_worker = (td_error_worker ** 2).sum() / mask.sum() + q_attend_regs_worker

        # --- B. 计算 Manager Loss ---
        # 1. Manager Mixer & Loss
        chosen_action_qvals_manager = th.gather(mac_out_worker[:, :-1], dim=3, index=actions).squeeze(3)
        chosen_action_qvals_manager, q_attend_regs_manager, _ = self.mixer_manager(chosen_action_qvals_manager, batch["state"][:, :-1], actions)
        
        with th.no_grad():
            target_max_qvals_manager = target_mac_out.max(dim=3)[0]
            # 这里必须重新获取 target_next_actions 或者复用，确保维度匹配
            if 'target_next_actions' not in locals():
                target_next_actions = target_mac_out.max(dim=3, keepdim=True)[1]
            
            target_max_qvals_manager, _, _ = self.target_mixer_manager(target_max_qvals_manager, batch["state"][:, 1:], target_next_actions)
            targets_manager = rewards + self.args.gamma * (1 - terminated) * target_max_qvals_manager

        td_error_manager = (chosen_action_qvals_manager - targets_manager.detach())
        loss_manager = (td_error_manager ** 2).sum() / mask.sum() + q_attend_regs_manager


        # ===================================================
        # 2. 反向传播与梯度处理 (关键修复)
        # ===================================================

        # --- Step 1: Worker Backward ---
        self.worker_optimiser.zero_grad()
        # retain_graph=True 因为 Manager 还要用到这个计算图
        loss_worker.backward(retain_graph=True) 
        
        # [关键] 缓存 Worker 的梯度！
        # 因为接下来运行 loss_manager.backward() 时，Manager 的梯度会流经 Worker，
        # 污染 Worker 的梯度。我们需要保存只有 Intrinsic Reward 产生的纯净梯度。
        worker_grads = [p.grad.clone() if p.grad is not None else None for p in self.worker_params]

        # --- Step 2: Manager Backward ---
        self.manager_optimiser.zero_grad()
        loss_manager.backward()
        # 此时 Manager 的参数 (Goal Agent) 已经有了正确的梯度
        # Worker 的参数也有了梯度（但这是来自 Manager 的，我们不要）

        # --- Step 3: 应用梯度与更新 ---
        
        # 更新 Manager (Goal Agent)
        grad_norm_manager = th.nn.utils.clip_grad_norm_(self.manager_params, self.args.grad_norm_clip)
        self.manager_optimiser.step()

        # 还原 Worker 的纯净梯度
        for p, g in zip(self.worker_params, worker_grads):
            if g is not None:
                p.grad = g # 覆盖掉刚才 Manager 反传过来的梯度
        
        # 更新 Worker
        grad_norm_worker = th.nn.utils.clip_grad_norm_(self.worker_params, self.args.grad_norm_clip)
        self.worker_optimiser.step()

        # ===================================================
        # LOGGING
        # ===================================================
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_worker", loss_worker.item(), t_env)
            self.logger.log_stat("loss_manager", loss_manager.item(), t_env)
            self.logger.log_stat("intrinsic_reward_mean", intrinsic_rewards.mean().item(), t_env)
            self.logger.log_stat("grad_norm_worker", grad_norm_worker, t_env)
            self.logger.log_stat("grad_norm_manager", grad_norm_manager, t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer_worker.load_state_dict(self.mixer_worker.state_dict())
        self.target_mixer_manager.load_state_dict(self.mixer_manager.state_dict())
        self.logger.console_logger.info("Updated target networks")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.state_embed.cuda()
        self.mixer_worker.cuda()
        self.mixer_manager.cuda()
        self.target_mixer_worker.cuda()
        self.target_mixer_manager.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mixer_worker.state_dict(), "{}/mixer_worker.th".format(path))
        th.save(self.mixer_manager.state_dict(), "{}/mixer_manager.th".format(path))
        th.save(self.worker_optimiser.state_dict(), "{}/opt_worker.th".format(path))
        th.save(self.manager_optimiser.state_dict(), "{}/opt_manager.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.mixer_worker.load_state_dict(th.load("{}/mixer_worker.th".format(path), map_location=lambda storage, loc: storage))
        self.mixer_manager.load_state_dict(th.load("{}/mixer_manager.th".format(path), map_location=lambda storage, loc: storage))
        self.worker_optimiser.load_state_dict(th.load("{}/opt_worker.th".format(path), map_location=lambda storage, loc: storage))
        self.manager_optimiser.load_state_dict(th.load("{}/opt_manager.th".format(path), map_location=lambda storage, loc: storage))