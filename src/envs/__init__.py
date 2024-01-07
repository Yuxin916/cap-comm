from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
from .gymmawrapper import _GymmaWrapper

# this function builds a class env (type MultiAgentEnv) with keyworld argument **kwargs.
def env_fn(env, **kwargs) -> MultiAgentEnv:
    # env_fn函数的输入是env和**kwargs，输出是MultiAgentEnv类的对象env(**kwargs)
    # 进入_GymmaWrapper类的__init__函数
    return env(**kwargs)


REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )

# 新环境的注册
REGISTRY["gymma"] = partial(env_fn, env=_GymmaWrapper)
