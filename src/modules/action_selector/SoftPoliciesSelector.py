import torch as th
from torch.distributions import Categorical


class SoftPoliciesSelector:
    """
    在决策时考虑多种可能动作的策略，而不是总是选择最优动作
    """

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # agent_inputs: [env_num, agent_num, action_num概率]
        # 这里的env_num是指还没有terminated的环境

        # Categorical 通常用于表示具有给定概率的离散动作空间
        m = Categorical(agent_inputs)

        # 从上述创建的分类分布中采样动作 .sample() 方法从分布中随机选择动作，
        # .long() 是将结果转换为长整型
        picked_actions = m.sample().long()
        # 【env_num,agent_num】
        return picked_actions
