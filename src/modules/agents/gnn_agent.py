import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class GNNAgent(torch.nn.Module):
    def __init__(self, input_shape, args, training=True):
        super(GNNAgent, self).__init__()
        """
        input_shape: (int) the shape of the obs to one agent
        args: (argparse) the arguments of the experiment
        """
        self.args = args

        self.training = training
        self.input_shape = input_shape
        # GNN有几层
        self.message_passes = self.args.num_layers

        # 默认gnn输入包含capability
        self.capabilities_skip_gnn = False
        # 如果 args 中有 capabilities_skip_gnn 属性且为真
        if hasattr(self.args, "capabilities_skip_gnn"):
            if self.args.capabilities_skip_gnn:
                self.capabilities_skip_gnn = True
                # input_shape 将减去 args 中定义的 capability_shape
                input_shape = input_shape - self.args.capability_shape

        # MLP输入维度到hidden维度
        self.encoder = nn.Sequential(nn.Linear(input_shape, self.args.hidden_dim),
                                      nn.ReLU(inplace=True))

        # 是否使用图注意力机制
        if self.args.use_graph_attention:
            self.messages = nn.MultiHeadAttention(n_heads=self.args.n_heads,
                                                  input_dim=self.args.hidden_dim,
                                                  embed_dim=self.embed_dim)
        else:
            self.messages = nn.Sequential(nn.Linear(self.args.msg_hidden_dim, self.args.msg_hidden_dim, bias=False),
                                          nn.ReLU(inplace=True),
                                          nn.Linear(self.args.msg_hidden_dim, self.args.msg_hidden_dim, bias=False))
        # 计算策略头的输入维度
        policy_head_input_shape = 0

        if self.capabilities_skip_gnn:
            policy_head_input_shape += self.args.capability_shape

        if self.message_passes > 0:
            policy_head_input_shape += self.args.msg_hidden_dim + self.args.hidden_dim
        else:
            policy_head_input_shape += self.args.hidden_dim

        # 策略layer
        self.policy_head = nn.Sequential(nn.Linear(policy_head_input_shape, self.args.hidden_dim),
                                         nn.ReLU(inplace=True))

        # 动作输出 - 根据策略头的输出生成动作
        self.actions = nn.Linear(self.args.hidden_dim, self.args.n_actions)

    def cuda_transfer(self):
        for i in range(self.args.num_layers):
            self.convs[i].cuda()
            print('\n\n\n\n\nTRANFERED TO GPUUUU')

    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(self.args.hidden_dim) #self.encoder.weight.new(1, self.args.hidden_dim).zero_()

    def calc_adjacency_hat(self, adj_matrix):
        """
        Calculates the normalized adjacency matrix including self-loops.
        This bounds the eigenv values and repeated applications of this graph
        shift operator could lead to numerical instability if this is not done, as
        well as exploding/vanishing gradients.
        """
        # This adds a self-loop so that a nodes own message is passed onto itself
        # 节点与自己的连接 - 允许每个节点在消息传递过程中考虑自己的特征
        A_hat = (adj_matrix + torch.eye(adj_matrix.shape[-1])).squeeze()#.to(self.device) 

        #
        D_hat = torch.pow(A_hat.sum(1), -0.5).unsqueeze(-1) * torch.ones(A_hat.shape)

        return torch.matmul(torch.matmul(D_hat, A_hat), D_hat)

    def forward(self, x, adj_matrix):
        """
        x: (torch.tensor) the input to the agent  (环境数量*n_agents, obs_dim)
        adj_matrix: (torch.tensor) the adjacency matrix of the graph  (环境数量, n_agents, n_agents)
        """
        if self.capabilities_skip_gnn:
            # 将输入数据 x 中与capability相关的部分分离出来
            capabilities = x[:, -self.args.capability_shape:]
            x = x[:, :-self.args.capability_shape]

        # Get the Normalized adjacency matrix, and add self-loops
        # 规范化邻接矩阵，添加自环
        if self.args.normalize_adj_matrix:
            comm_mat = self.calc_adjacency_hat(adj_matrix)
        else:
            comm_mat = adj_matrix.float()
        # comm_mat - {batch_size, N, N}

        # 对输入数据 x 进行编码，生成每个node的嵌入表示 [(环境数量*n_agents,self.hidden_dim)]
        enc = self.encoder(x)

        # reshape encoder output (环境数量, n_agents, self.hidden_dim)
        msg = enc.view(-1, self.args.n_agents, self.args.hidden_dim)

        # 消息传递不是一次性的，而是重复进行多次
        for k in range(self.message_passes):
            # comm_mat 代表的是邻接矩阵，用于表示图中节点之间的连接关系
            # msg 代表的是代表每个节点的特征 - 被整理成适合进行消息传递的形状
            # 输出的msg代表每个节点融合了邻居信息后的新表示 【环境数量，n_agents, self.hidden_dim】
            msg = self.messages(torch.matmul(comm_mat, msg))

        # 输入给policy head, shape is (环境数量 * n_agents, hidden_dim)
        msg = msg.view(-1, self.args.hidden_dim)

        # 检查是否有消息传递层被使用
        if self.message_passes > 0:
            # 将编码器的输出 (enc) 和消息传递后的输出 (msg) 沿着最后一个维度（dim=-1）拼接起来
            # 为了结合原始节点特征（由编码器 enc 提供）和经过图结构处理的特征（由 msg 提供）
            h = torch.cat((enc, msg), dim=-1)
        else:
            # 如果没有进行消息传递（即 self.message_passes 等于 0），则直接使用编码器的输出作为最终特征表示。
            # 这意味着在这种情况下，模型只考虑原始特征，而不涉及基于图结构的特征加工
            h = enc

        if self.capabilities_skip_gnn:
            h = torch.cat((h, capabilities), dim=-1)

        # 将最终的特征表示输入给策略头，生成动作
        h = self.policy_head(h)
        # 生成动作 - (环境数量 * n_agents, n_actions)
        actions = self.actions(h)

        return actions, h
    
class DualChannelGNNAgent(torch.nn.Module):
    """
    Dual Channel gnn (Agents that have two types of observations which they want to learn to communicate separately)
    """
    def __init__(self, input_shape, args, training=True):
        """
        """
        super(DualChannelGNNAgent, self).__init__()
        self.args = args
        self.capability_shape = self.args.capability_shape
        self.input_shape_a = input_shape - self.capability_shape
        self.channel_A = GNNAgent(self.input_shape_a, args=args, training=training)
        self.channel_B = GNNAgent(self.capability_shape, args=args, training=training)

        self.actions = nn.Linear(2*self.args.hidden_dim, self.args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        self.channel_A.init_hidden()
        self.channel_B.init_hidden()
        return torch.zeros(self.args.hidden_dim) #self.encoder.weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, x, adj_matrix):
        """
        Forward the two inputs to each channel and adjacency matrix
        through the model

        params:
            x tensor
        returns:
            action (tensor)
            h (tensor) : concatenation of the two gnn outputs.
        """
        x_a = x[:, :self.input_shape_a]
        x_b = x[:, self.input_shape_a:] # should be the capabilities
        _, h_a = self.channel_A(x_a, adj_matrix)
        _, h_b = self.channel_B(x_b, adj_matrix)

        h = torch.concat((h_a, h_b), dim=-1)
        action = self.actions(h)
        return(action, h)