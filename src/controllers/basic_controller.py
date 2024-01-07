from modules.agents import REGISTRY as agent_REGISTRY
from modules.action_selector import REGISTRY as action_REGISTRY
from communication import REGISTRY as comm_REGISTRY
from modules.critics import REGISTRY as critic_resigtry
import torch as th
from torchsummary import summary


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        # 单个agent的输入维度
        input_shape = self._get_input_shape(scheme)
        # 这里的scheme包含已经在episode_buffer更新的信息
        # state,obs,actions,adj_matrix,avail_actions,reward,terminated,action_onehot,filled信息
        self.scheme = scheme

        # 把要通信的信息处理网络独立于actor网络，因此actor和critic在ippo的时候输入大小一致
        # 只在ippo的时候使用
        if self.args.separated_policy:
            self._build_comm(input_shape)
            self._build_agents(self.args.msg_out_size)
            # build the critic
            self._build_critic(self.args.msg_out_size)
        else:
            self._build_agents(input_shape)

        # actor输出的形式 - pi_logits
        self.agent_output_type = args.agent_output_type

        # 从策略选择动作的方式 - greedy或者soft或者multimodal
        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """
        ep_batch: 所有环境的一个episode的transition data 【env_num,episode_limit,*shape】
        t_ep:
        t_env:
        bs: 还没有结束的环境的index
        test_mode: 是否是测试模式
        """
        # Only select actions for the selected batch elements in bs

        # 所有环境在t_ep时刻的每个agent的可用动作 [env_num,agent_num,action_num]
        avail_actions = ep_batch["avail_actions"][:, t_ep]

        # softmax后的 [env_num, agent_num, action_num]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)

        # 根据概率分别选择specific env thread的动作
        chosen_actions = self.action_selector.select_action(agent_outputs[bs],
                                                            avail_actions[bs],
                                                            t_env,
                                                            test_mode=test_mode)
        # 没有terminate的环境里所有的agent的动作 离散动作分类 [env_num, agent_num]
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):

        # Build the communication outputs from the gnn, which are inputs to both actor and critic
        # 只适用于ippo的情况
        if self.args.separated_policy:
            comm_input = self._build_inputs(ep_batch, t)
            agent_inputs = self._build_msg(comm_input, ep_batch.batch_size, ep_batch["adj_matrix"][:, t, ...], ep_batch.device)# self._build_inputs(ep_batch, t)
        else:
            # 在t时刻 (环境数量*n_agents, obs_dim) # TODO：为什么agent的actor可以拿到所有agent的obs
            agent_inputs = self._build_inputs(ep_batch, t)

        # 所有环境在t_ep时刻的每个agent的可用动作 [env_num,agent_num,action_num]
        avail_actions = ep_batch["avail_actions"][:, t]

        # use this if just the actor is a gnn.
        if self.args.agent == "gnn" or self.args.agent == "gat" or \
                self.args.agent == "dual_channel_gnn" or self.args.agent == "dual_channel_gat":
            # actor的输入是t时刻所有环境下的
            # agent_inputs (环境数量*n_agents, obs_dim)
            # adj_matrix (环境数量, n_agents, n_agents)
            agent_outs, _ = self.agent(agent_inputs, ep_batch["adj_matrix"][:, t, ...])
        
        # use gnn has the gnn as the comm layer, and mlps for both actor and critic. (like GPPO)
        elif self.args.use_gnn:
            agent_outs, _ = self.agent(agent_inputs)
        else:
            # 生成动作 - (环境数量 * n_agents, n_actions)
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            # 给agent out的最后一个动作维度做softmax (环境数量 * n_agents, n_actions[softmax])
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

            if self.args.use_gnn:
                if not test_mode:
                    epsilon_action_num = agent_outs[-1]

                    if getattr(self.args, "mask_before_softmax", True):
                        # select random action with probability epsilon
                        epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).floor()

                    agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                                        + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)
            
                    if getattr(self.args, "mask_before_softmax", True):
                        # Zero out the unavailable actions
                        agent_outs[reshaped_avail_actions == 0] = 0.0

        # reshape agent_outs to (环境数量, n_agents, n_actions)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        # 根据batch_size初始化hidden_states
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        # 返回actor的参数
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()
        if self.args.use_gnn:
            self.gnn.cuda_transfer()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))
        th.save(self.agent, "{}/agent_model.pt".format(path))
        if self.args.separated_policy:
            th.save(self.gnn.state_dict(), "{}/gnn.th".format(path))
            th.save(self.agent, "{}/gnn_model.pt".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        if self.args.separated_policy:
            self.gnn.load_state_dict(th.load("{}/gnn.th".format(path), map_location=lambda storage, loc: storage))
  
    def _build_agents(self, input_shape):
        """
        Agent is the actor portion of the policy
        """
        # actor的网络类型 在算法yaml中定义
        print("\033[31m" + self.args.agent + "\033[0m")
        # 根据agent类型，选择rnn或者mlp或者gnn或者gat并且初始化
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        # summary(self.agent, input_size=input_shape)
        print("\033[31m" + str(type(self.agent)) + "\033[0m")

    def _build_critic(self, input_shape):
        """
        Critic network
        """
        print("\033[31m" + self.args.critic_type + "\033[0m")
        # 根据critic类型，选择coma或者maddpg或者centralized_critic或者MAAC
        self.critic = critic_resigtry[self.args.critic_type](self.scheme, self.args)

    def _build_comm(self, input_shape):
        # Build the communication network
        # 可以选择gcn或者gat或者gnn
        self.gnn = comm_REGISTRY["gcn"](input_shape, self.args)
        # self.gnn = agent_REGISTRY["gnn"](input_shape, self.args)

    def _build_msg(self, batch, batch_size, adj_matrix, device):
        """
        If the gnn is used to feed to both the actor and critic, it goes here
        """
        input_observation = batch.reshape(batch_size, self.n_agents, -1).to(device=device)
        msg_enc = self.gnn(input_observation, th.tensor(adj_matrix, device=device))
        reshaped_msg_enc = msg_enc.reshape(batch_size * self.n_agents, -1)
        return reshaped_msg_enc

    def _build_inputs(self, batch, t):
        """Assumes homogenous agents with flat observations.
        Other MACs might want to e.g. delegate building inputs to each agent"""

        # 环境数量
        bs = batch.batch_size
        inputs = []
        # one list, a tensor with dim (环境数量, n_agents, obs_dim)
        # 假设是homogenous agents，obs是平坦的
        inputs.append(batch["obs"][:, t])  # b1av
        # 是否在obs中加入last_action
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        # 是否在obs中加入agent_id
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        # reshape成(环境数量*n_agents, obs_dim) - 在t时刻的obs
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)

        return inputs

    def _get_input_shape(self, scheme):
        # 获取actor的输入维度
        input_shape = scheme["obs"]["vshape"]
        # 是否在obs中加入last_action
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        # 是否在obs中加入agent_id
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
