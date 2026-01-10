import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.qatten_new import QattenMixer_New
import torch as th
from torch.optim import RMSprop
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np

class FeudalLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.params = list(mac.parameters()) 

        if args.mixer == "qatten_new":
            self.mixer_worker = QattenMixer_New(args)
            self.mixer_manager = QattenMixer_New(args)
        else:
            raise ValueError("Feudal Learner strictly requires QAtten_New mixer")

        self.target_mixer_worker = copy.deepcopy(self.mixer_worker)
        self.target_mixer_manager = copy.deepcopy(self.mixer_manager)

        self.worker_params = list(mac.agent.parameters()) + list(self.mixer_worker.parameters())
        self.worker_optimiser = RMSprop(params=self.worker_params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.manager_params = list(mac.goal_agent.parameters()) + list(self.mixer_manager.parameters())
        self.manager_optimiser = RMSprop(params=self.manager_params, lr=args.manager_lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.target_mac = copy.deepcopy(mac)
        self.last_target_update_episode = 0

        obs_dim = scheme["obs"]["vshape"]
        self.state_embed = th.nn.Linear(obs_dim, args.goal_dim).to(args.device)
        self.worker_optimiser.add_param_group({'params': self.state_embed.parameters()})
        
        self.entropy_coef = getattr(args, "entropy_coef", 0.01)

        # State Normalization
        state_shape = scheme["state"]["vshape"]
        if isinstance(state_shape, int):
            state_shape = (state_shape,)
        self.state_norm = nn.LayerNorm(state_shape).to(args.device)
        self.manager_optimiser.add_param_group({'params': self.state_norm.parameters()})

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        goals = batch["goals"][:, :-1]

        # State Normalization
        state = batch["state"]
        normalized_state = self.state_norm(state)
        state_curr = normalized_state[:, :-1]
        state_next = normalized_state[:, 1:]

        # ===================================================
        # 1. 前向传播与 Loss 计算
        # ===================================================
        
        # --- A. Worker Calculation ---
        with th.no_grad():
            obs_curr = batch["obs"][:, :-1]
            obs_next = batch["obs"][:, 1:]
            diff = obs_next - obs_curr
            diff_embed = self.state_embed(diff)
            detached_goals = goals.detach()
            
            # [关键修复] 手动计算 Cosine Similarity，增加数值稳定性
            # 防止 diff_embed 模长为 0 导致梯度爆炸
            eps = 1e-5 # 比 1e-8 更大的 eps，确保分母安全
            norm_diff = diff_embed.norm(dim=-1, keepdim=True) + eps
            norm_goals = detached_goals.norm(dim=-1, keepdim=True) + eps
            
            # (A . B) / (|A| * |B|)
            dot_product = (diff_embed * detached_goals).sum(dim=-1, keepdim=True)
            intrinsic_rewards = dot_product / (norm_diff * norm_goals)
            
            intrinsic_rewards = intrinsic_rewards / 10.0

        mac_out_worker = []
        manager_logits = [] 
        self.mac.init_hidden(batch.batch_size)
        
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out_worker.append(agent_outs)
            if hasattr(self.mac, 'current_logits') and self.mac.current_logits is not None:
                manager_logits.append(self.mac.current_logits)
            else:
                pass 

        mac_out_worker = th.stack(mac_out_worker, dim=1)

        chosen_action_qvals_worker = th.gather(mac_out_worker[:, :-1], dim=3, index=actions).squeeze(3)
        chosen_action_qvals_worker, q_attend_regs_worker, _ = self.mixer_worker(chosen_action_qvals_worker, state_curr, actions)

        # 防止 mask.sum() 为 0 (虽然极少见)
        mask_sum = mask.sum()
        if mask_sum < 1.0:
            mask_sum = 1.0 # 避免除以 0

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
            
            target_max_qvals_worker, _, _ = self.target_mixer_worker(target_max_qvals_worker, state_next, target_next_actions)

        targets_worker = intrinsic_rewards.sum(dim=2) + self.args.gamma * (1 - terminated) * target_max_qvals_worker
        td_error_worker = (chosen_action_qvals_worker - targets_worker.detach())
        loss_worker = (td_error_worker ** 2).sum() / mask_sum + q_attend_regs_worker

        # --- B. Manager Calculation ---
        chosen_action_qvals_manager = th.gather(mac_out_worker[:, :-1], dim=3, index=actions).squeeze(3)
        chosen_action_qvals_manager, q_attend_regs_manager, _ = self.mixer_manager(chosen_action_qvals_manager, state_curr, actions)
        
        with th.no_grad():
            target_max_qvals_manager = target_mac_out.max(dim=3)[0]
            if 'target_next_actions' not in locals():
                target_next_actions = target_mac_out.max(dim=3, keepdim=True)[1]
            
            target_max_qvals_manager, _, _ = self.target_mixer_manager(target_max_qvals_manager, state_next, target_next_actions)
            targets_manager = rewards + self.args.gamma * (1 - terminated) * target_max_qvals_manager


        # Entropy Reg
        entropy_loss = 0
        if len(manager_logits) > 0:
            T = mask.shape[1]
            stacked_logits = th.stack(manager_logits[:T], dim=1)
            probs = F.softmax(stacked_logits, dim=-1)
            log_probs = F.log_softmax(stacked_logits, dim=-1)
            entropy = - (probs * log_probs).sum(dim=-1) 
            entropy = entropy.view(batch.batch_size, self.args.n_agents, -1)
            entropy = entropy.transpose(1, 2) 
            masked_entropy = entropy * mask
            entropy_loss = masked_entropy.sum() / (mask_sum * self.args.n_agents)


        td_error_manager = (chosen_action_qvals_manager - targets_manager.detach())
        loss_manager = (td_error_manager ** 2).sum() / mask_sum + q_attend_regs_manager - self.entropy_coef * entropy_loss

        # ===================================================
        # 2. 反向传播
        # ===================================================

        self.worker_optimiser.zero_grad()
        loss_worker.backward(retain_graph=True) 
        worker_grads = [p.grad.clone() if p.grad is not None else None for p in self.worker_params]

        self.manager_optimiser.zero_grad()
        loss_manager.backward()

        grad_norm_manager = th.nn.utils.clip_grad_norm_(self.manager_params, self.args.grad_norm_clip)
        self.manager_optimiser.step()

        for p, g in zip(self.worker_params, worker_grads):
            if g is not None:
                p.grad = g
        grad_norm_worker = th.nn.utils.clip_grad_norm_(self.worker_params, self.args.grad_norm_clip)
        self.worker_optimiser.step()

        # LOGGING
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_worker", loss_worker.item(), t_env)
            self.logger.log_stat("loss_manager", loss_manager.item(), t_env)
            self.logger.log_stat("intrinsic_reward_mean", intrinsic_rewards.mean().item(), t_env)
            self.logger.log_stat("manager_entropy", entropy_loss.item() if isinstance(entropy_loss, th.Tensor) else 0, t_env)
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