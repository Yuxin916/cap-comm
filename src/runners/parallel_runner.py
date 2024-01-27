from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
from .message_env import CloudpickleWrapper, env_worker

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        # 这里的args读取的是
        #     - 算法.yaml,
        #     - default.yaml
        #     - gymma.yaml文件
        # 中的参数，还包含命令行传入/override的参数
        self.args = args
        # 这里的args.env_args读取的是main文件中的随机生成的seed
        self.args.env_args["seed"] = self.args.seed
        # logger是sacred的logger
        self.logger = logger
        # 并行环境的数量
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        # 这里的parent_conns和worker_conns是两个长度为env thread的元组，每个元素都是一个Pipe对象
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        # 进入env的init函数, 因为这里的self.args.env是外接环境，也就是gymma
        # 新增的环境通过gymma进行注册
        # REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
        env_fn = env_REGISTRY[self.args.env]
        # 从args中单独提取出所有的环境参数env_args，生成多进程的环境参数env_args
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        # 对于每一个进程，都将env_args中的seed加上一个偏移量
        for i in range(self.batch_size):
            env_args[i]["seed"] += i
        # 这里的self.ps是一个长度为batch_size的列表，每个元素都是一个Process对象
        # Process是multiprocessing中的一个类，用于创建进程
        # target是进程执行的函数，args是传递给target的参数
        # 这里的target是env_worker，args是worker_conn和CloudpickleWrapper(partial(env_fn, **env_arg))
        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))))
                            for env_arg, worker_conn in zip(env_args, self.worker_conns)]
        # 将所有的进程都启动
        for p in self.ps:
            p.daemon = True
            # 这里的p.start()是调用Process类中的start方法，该方法会调用target函数 - env_worker 去看这个文件里面的env_worker函数
            p.start()

        # 发送get_env_info消息给每一个env，去看env_worker函数，得到env_info
        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        # 用于追踪多个并行环境中的当前时间步
        self.t = 0
        # 在这些环境中累积的时间步总数
        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        self.log_train_stats_t = -100000

    def setup(self, scheme, groups, preprocess, mac):
        # 这里的self.new_batch是一个函数，用于创建EpisodeBatch对象
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        # 在初始化的时候，已经发送了get_env_info消息给第一个env，这里直接返回env_info
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        # 发送close消息给每一个env
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        """
        重置batch
        batch.data.transition_data是一个字典，包含了all np.zeros

        state [env_thread, max_episode_len, n_agents * obs_dim]
        obs [env_thread, max_episode_len, n_agents, obs_dim]
        action [env_thread, max_episode_len, n_agents, 1]
        adj_matrix [env_thread, max_episode_len, n_agents, n_agents]
        avail_actions [env_thread, max_episode_len, n_agents, action_dim]
        reward [env_thread, max_episode_len, 1]
        terminated [env_thread, max_episode_len, 1]
        action_onehot [env_thread, max_episode_len, n_agents, action_dim]
        filled [env_thread, max_episode_len, 1]
        """
        self.batch = self.new_batch()

        # reset所有环境
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
            "adj_matrix":[]

        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            # 接收reset的返回值
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            pre_transition_data["adj_matrix"].append(data["adj_matrix"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0

    def run(self, test_mode=False):
        # reset所有环境以及一个batch的episode data
        self.reset()

        all_terminated = False

        # batch_size是并行环境的数量
        # 初始化episode_returns和episode_lengths [对所有环境]
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]

        # 初始化actor的hidden state [对所有环境]
        self.mac.init_hidden(batch_size=self.batch_size)

        # 刚开始所有的并行环境都没有结束
        terminated = [False for _ in range(self.batch_size)]
        # 只包含没有terminated的环境的index [list of int]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION

        while True:
            # 直到所有环境都结束就break

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.mac.select_actions(self.batch, t_ep=self.t,
                                              t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            # 把actions转换成numpy 【环境数，智能体数量】
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            # 把选择的actions放入batch中 更新batch.data.transition_data里的action
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            # 把actions发送给每一个env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                # 如果这个env没有terminated
                if idx in envs_not_terminated:  # We produced actions for this env
                    if not terminated[idx]:  # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                    action_idx += 1  # actions is not a list over every env

            # 更新还没有terminated的环境的index Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]

            # 如果所有的env都terminated了，就break
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [],
                "adj_matrix": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    # 记录第idx个环境的episode_returns和episode_lengths
                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1

                    if not test_mode:
                        # 包含了所有环境总共跑了多少个timestep
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        # 如果这个环境结束了，就把info放入final_env_infos,代表这个episode最后的info
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    pre_transition_data["adj_matrix"].append(data["adj_matrix"])

            # Add post_transiton data into the batch
            # 属于这个时刻的reward和terminated
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            # 下一个时刻的state，avail_actions，obs，adj_matrix
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            # 在所有环境里的累积timestep
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        # 把所有的env的stats都收集起来 but是空集合啊...
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        # 分离train和test的stats和returns
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""

        infos = [cur_stats] + final_env_infos

        # 合并了所有环境的info 在cur_stats里，包含total_overlap，edge_count和violation_occurred
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})

        # 给cur_stats添加 n_episodes（总episode数量 环境数量）和 ep_length（所有环境的episode分别多少timestep的总和）
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        # 把所有环境的episode_returns都放入cur_returns
        cur_returns.extend(episode_returns)

        # 分离train和test的stats和returns
        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()




