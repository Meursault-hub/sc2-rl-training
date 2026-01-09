from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import time


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = {"reward": [], "delta_enemy": [], "delta_deaths": [], "delta_ally": []}
        self.test_returns = {"reward": [], "delta_enemy": [], "delta_deaths": [], "delta_ally": []}
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = {"reward": 0, "delta_enemy": 0, "delta_deaths": 0, "delta_ally": 0}
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])

            # === 【新增】在这里插入慢放逻辑 ===
            # 检查是否开启了渲染模式 (命令行传入了 env_args.render=True)
            if self.args.env_args.get("render", False):
                time.sleep(0.5)  # 休眠 0.2 秒 (可以改成 0.1 或 0.5)
            # ================================

            reward, delta_enemy, delta_deaths, delta_ally = reward
            episode_return["reward"] += reward
            episode_return["delta_enemy"] += delta_enemy
            episode_return["delta_deaths"] += delta_deaths
            episode_return["delta_ally"] += delta_ally

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns["reward"].append(episode_return["reward"])
        cur_returns["delta_enemy"].append(episode_return["delta_enemy"])
        cur_returns["delta_deaths"].append(episode_return["delta_deaths"])
        cur_returns["delta_ally"].append(episode_return["delta_ally"])

        if test_mode and (len(self.test_returns["reward"]) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "reward_mean", float(np.mean(returns["reward"])), self.t_env)
        self.logger.log_stat(prefix + "reward_std", float(np.std(returns["reward"])), self.t_env)
        self.logger.log_stat(prefix + "delta_enemy_mean", float(np.mean(returns["delta_enemy"])), self.t_env)
        self.logger.log_stat(prefix + "delta_deaths_mean", float(np.mean(returns["delta_deaths"])), self.t_env)
        self.logger.log_stat(prefix + "delta_ally_mean", float(np.mean(returns["delta_ally"])), self.t_env)
        for k in returns:
            returns[k].clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
