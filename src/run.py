import datetime
from functools import partial
from math import ceil
import numpy as np
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

def run(_run, _config, _log, pymongo_client=None):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    if pymongo_client is not None:
        print("Attempting to close mongodb client")
        pymongo_client.close()
        print("Mongodb client closed")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(0)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.unit_dim = env_info["unit_dim"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return
        
        # === [修改] 优先判断：当前路径是否直接就是模型文件夹？ ===
        # 检查目录下是否存在 'agent.th'，如果存在，说明这本身就是模型目录
        if os.path.exists(os.path.join(args.checkpoint_path, "agent.th")):
            model_path = args.checkpoint_path
            logger.console_logger.info("Loading model directly from {}".format(model_path))
            learner.load_models(model_path)
            # 尝试从路径中提取步数用于日志显示（非必须）
            runner.t_env = 0 
        # =======================================================

        else :
            # Go through all files in args.checkpoint_path
            for name in os.listdir(args.checkpoint_path):
                full_name = os.path.join(args.checkpoint_path, name)
                # Check if they are dirs the names of which are numbers
                if os.path.isdir(full_name) and name.isdigit():
                    timesteps.append(int(name))

            if args.load_step == 0:
                # choose the max timestep
                timestep_to_load = max(timesteps)
            else:
                # choose the timestep closest to load_step
                timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

            model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

            logger.console_logger.info("Loading model from {}".format(model_path))
            learner.load_models(model_path)
            runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # === 【修改 1】 在这里插入初始化代码 ===
    best_test_win_rate = -1.0
    best_test_reward = -float('inf')
    # ====================================   

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        # === [修改] 添加热身期逻辑 ===
        # 设定热身阈值，例如 1000 个 episode 
        # 这确保了在 buffer 只有很少数据时，不会过早开始训练导致过拟合
        WARMUP_EPISODES = 500

        if buffer.can_sample(args.batch_size) and buffer.episodes_in_buffer > WARMUP_EPISODES:
            for _ in range(args.training_iters):
                episode_sample = buffer.sample(args.batch_size)

                # Truncate batch to only filled timesteps
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

            # === 【修改 2】 在这里插入保存逻辑 ===
            # 从 logger 中获取最新的测试数据
            # 注意：PyMARL 的 logger.stats 存储格式为 {key: [(step, value), ...]}
            # 我们取列表的最后一个元素 [-1] 的值 [1]
            
            # 安全获取当前胜率 (有些地图可能没有 win_rate，给个默认值 0)
            if "test_battle_won_mean" in logger.stats:
                curr_test_win_rate = logger.stats["test_battle_won_mean"][-1][1]
            else:
                curr_test_win_rate = 0.0

            # 安全获取当前奖励
            if "test_reward_mean" in logger.stats:
                curr_test_reward = logger.stats["test_reward_mean"][-1][1]
            else:
                curr_test_reward = -float('inf')

            # 判定逻辑：
            # 1. 只有胜率达到 100% (>= 1.0) 才考虑保存 "Best Model"
            # 2. 如果是新的 100% 且 奖励比之前的最佳奖励更高，则更新保存
            
            if curr_test_win_rate >= 1.0:
                is_new_best = False
                
                # 情况 A: 第一次达到 100% (之前的最佳胜率不到 1.0)
                if best_test_win_rate < 1.0:
                    is_new_best = True
                # 情况 B: 之前已经是 100%，但这次的奖励更高
                elif curr_test_reward > best_test_reward:
                    is_new_best = True
                
                if is_new_best:
                    # 更新最佳记录
                    best_test_win_rate = curr_test_win_rate
                    best_test_reward = curr_test_reward
                    
                    # 构造保存路径：results/models/token/best_model
                    save_path = os.path.join(args.local_results_path, "models", args.unique_token, "best_model")
                    os.makedirs(save_path, exist_ok=True)
                    
                    logger.console_logger.info("🚀 [New Best Model] Win Rate: {:.2%} | Reward: {:.4f}".format(curr_test_win_rate, curr_test_reward))
                    logger.console_logger.info("💾 Saving best model to: {}".format(save_path))
                    
                    learner.save_models(save_path)
            # ====================================

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


# TODO: Clean this up
def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    # assert (config["run_mode"] in ["parallel_subproc"] and config["use_replay_buffer"]) or (not config["run_mode"] in ["parallel_subproc"]),  \
    #     "need to use replay buffer if running in parallel mode!"

    # assert not (not config["use_replay_buffer"] and (config["batch_size_run"]!=config["batch_size"]) ) , "if not using replay buffer, require batch_size and batch_size_run to be the same."

    # if config["learner"] == "coma":
    #    assert (config["run_mode"] in ["parallel_subproc"]  and config["batch_size_run"]==config["batch_size"]) or \
    #    (not config["run_mode"] in ["parallel_subproc"]  and not config["use_replay_buffer"]), \
    #        "cannot use replay buffer for coma, unless in parallel mode, when it needs to have exactly have size batch_size."

    return config
