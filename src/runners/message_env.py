def env_worker(remote, env_fn):
    # Make environment
    # 进入这里的时候，env_fn是一个函数，调用它可以得到一个环境 -》在env文件夹的__init__.py中
    env = env_fn.x()
    last_env_info = None
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # log info on change or termination
            if last_env_info != env_info or terminated:
                # if terminated: print("terminated")
                last_env_info = env_info
                # print("info", env_info)

            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            adj_matrix = env.get_adj_matrix()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                "adj_matrix": adj_matrix,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs(),
                "adj_matrix": env.get_adj_matrix()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            # 在env文件夹的__init__.py中_GymmaWrapper继承的MultiAgentEnv类中有get_env_info函数
            # 你可真难找
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)
