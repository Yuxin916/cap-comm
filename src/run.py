from copy import deepcopy
import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger, get_unique_dirname
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot

def run(_run, config, _log):
    # hack to be able to modify Sacred config
    _config = deepcopy(config)

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config, indent=4, width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # save tensorboard files to new dir
    try:
        map_name = _config["env_args"]["map_name"]
    except:
        map_name = _config["env_args"]["key"]   
    # unique_token = get_unique_dirname(_config['name'], map_name)
    unique_token = os.path.join(map_name, config["unique_token"])
    unique_token = os.path.join(unique_token, str(_run._id))
    args.unique_token = unique_token 
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    # os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def run_sequential(args, logger):
    """
    这里的args读取的是
        - 算法.yaml,
        - default.yaml
        - gymma.yaml文件
    中的参数，还包含命令行传入/override的参数
    """
    # Init runner so we can get env info 环境运行器 - 获取环境信息&多进程 - parallel runner
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # 从env_info中获取环境信息
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "adj_matrix": {"vshape":(args.n_agents,), "group":"agents", "dtype": th.int},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}

    # 给动作空间添加一个onehot的预处理
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    # 初始化buffer
    buffer = ReplayBuffer(
        scheme,  # scheme是一个字典，包含了state,obs,actions等如上信息
        groups,  # groups是一个字典，包含了agents的数量
        args.buffer_size, # buffer可以存储的episode数量
        env_info["episode_limit"] + 1,  # 每个episode的最大长度 (#TODO: 这里是1000,而不是80
        preprocess=preprocess,  # 预处理的动作onehot
        device="cpu" if args.buffer_cpu_only else args.device,  # buffer的存储设备
    )

    # mac有basic, no_share和maddpg三种 智能体控制器 #TODO：MAT的接入需要添加一个mac
    # 原文使用的是mappo的share版本，即basic_mac
    # 一个重要属性就是智能体对象mac.agent - actor网络
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme 环境运行器 初始化
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner 智能体学习器
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    # 加载模型
    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path)
            )
            return

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

        if args.restart_from_pretrained:
            runner.t_env = 0
        else:
            runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return

    # 开始训练
    episode = 0
    # 每隔一定步数进行一次测试
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    # 按照训练步数进行训练
    while runner.t_env <= args.t_max:

        # 运行一个episode,更新了log，返回了所有并行环境的episode batch
        episode_batch = runner.run(test_mode=False)

        # 把所有环境下的episode batch存入buffer
        buffer.insert_episode_batch(episode_batch)

        # 判断此刻buffer里面储存的episode数量是否足够进行训练，这里的args.batch_size是指拿来训练的episode数量
        if buffer.can_sample(args.batch_size):
            # 从buffer中采样出args.batch_size个episode
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            # 从episode_sample中把所有episode的长度截断到最长的episode长度 在这里从1001截断到了82
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            # 转移到设备上
            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            # 更新learner！
            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        # 每隔一定步数进行一次测试
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        # 每隔一定步数进行一次保存
        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        # 记录总共跑了多少个episode
        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
