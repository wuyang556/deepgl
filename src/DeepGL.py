# coding = utf-8
# python 3.6.7
# Created by wuyang at2019/3/19
import networkx as nx
import numpy as np


class GraphProcessing:
    """由文本中建立一个图，并将图中的节点类型由str转为int\n
    # path指的是数据文本的绝对路径\n
    # first_line指的是文本第一行的信息是节点数和边数信息，需要略过，默认为有，用1代表\n
    # graph_direct指的是新建图的方向性，默认值为0，代表无向图"""

    def __init__(self, path, graph_direct=0, first_line=1):
        self.path = path
        self.first_line = first_line
        self.graph_direct = graph_direct

    def convert_graph_nodes_label_from_str_to_int(self):
        """把从文本里读取的图数据的节点数据类型由str转换为int"""
        with open(self.path, "r") as file:
            if self.first_line == 1:
                file.readline()
            edges = np.loadtxt(file, dtype=int)
            file.close()
        # print(edges.shape)
        # 初始化数据集中的节点序号从0开始
        # TODO 需要分辨输入的节点序号是否从0开始　
        edges_list = []
        for i in range(edges.shape[0]):
            edges_list.append([edges[i, 0]-1, edges[i, 1]-1])
        return edges_list

    def new_graph(self):
        """新建图，默认为无向图，\n
            path指的是数据文本的绝对路径，\n"""
        if self.graph_direct == 0:
            graph = nx.Graph()
        else:
            graph = nx.DiGraph()
        graph.add_edges_from(self.convert_graph_nodes_label_from_str_to_int())
        return graph


class GetGraphMessage:
    """图信息类，得到图的节点数，边数，邻接矩阵，拉普拉斯矩阵"""
    def __init__(self,graph=nx.Graph):
        self.graph = graph
        self.number_of_node = nx.number_of_nodes(self.graph)
        self.number_of_edge = nx.number_of_edges(self.graph)
        self.A = np.array(nx.linalg.adjacency_matrix(self.graph).todense(), dtype=np.float32)
        self.L = np.array(nx.linalg.laplacian_matrix(self.graph).todense(), dtype=np.float32)


class GraphElementFeature:
    """获取图元素的基本特征"""

    def __init__(self, graph):
        self.graph = graph

    def get_nodes_1_hop_neighbors(self):
        """获取所有节点的1-hop相邻节点，\n
            并以字典类型返回，key值为节点序号，value为节点对应的所有1-hop节点集合"""
        nodes_list = nx.nodes(self.graph)
        one_hip_neighbor = {}
        for node in nodes_list:
            neighbor = []
            for node_neighbor in nx.neighbors(self.graph, node):
                neighbor.append(node_neighbor)
            one_hip_neighbor[node] = neighbor
        return one_hip_neighbor

    def get_nodes_2_hops_neighbors(self):
        """获取所有节点的2-hops相邻节点，\n
            并以字典类型返回，key值为节点序号，value为节点对应的所有2-hops节点集合"""
        one_hop_neighbor = self.get_nodes_1_hop_neighbors()
        nodes_list = nx.nodes(self.graph)
        edges_list = nx.edges(self.graph)
        two_hops_neighbor = {}
        for node in nodes_list:
            neighbor = []
            for one_order_neighbor in one_hop_neighbor[node]:
                for edge in edges_list:
                    if edge[0] == one_order_neighbor and edge[1] != node:
                        # 保证找到的相邻节点不在1-hop相邻节点集合里
                        if edge[1] not in neighbor and edge[1] not in one_hop_neighbor[node]:
                            neighbor.append(edge[1])
                    elif edge[1] == one_order_neighbor and edge[0] != node:
                        # 保证找到的相邻节点不在1-hop相邻节点集合里
                        if edge[0] not in neighbor and edge[0] not in one_hop_neighbor[node]:
                            neighbor.append(edge[0])
            two_hops_neighbor[node] = neighbor
        return two_hops_neighbor

    def get_nodes_3_hops_neighbors(self):
        """获取所有节点的3-hops相邻节点，\n
            并以字典类型返回，key值为节点序号，value为节点对应的所有3-hops节点集合"""
        one_hop_neighbor = self.get_nodes_1_hop_neighbors()
        two_hops_neighbor = self.get_nodes_2_hops_neighbors()
        nodes_list = nx.nodes(self.graph)
        edges_list = nx.edges(self.graph)
        three_hops_neighbor = {}
        for node in nodes_list:
            neighbor = []
            for two_order_neighbor in two_hops_neighbor[node]:
                for edge in edges_list:
                    if edge[0] == two_order_neighbor and edge[1] != node:
                        # 保证找到的相邻节点不在2-hops相邻节点集合里
                        if edge[1] not in neighbor and edge[1] not in two_hops_neighbor[node]:
                            if edge[1] not in one_hop_neighbor[node]:
                                neighbor.append(edge[1])
                    if edge[1] == two_order_neighbor and edge[0] != node:
                        # 保证找到的相邻节点不在2-hops相邻节点集合里
                        if edge[0] not in neighbor and edge[0] not in two_hops_neighbor[node]:
                            if edge[0] not in one_hop_neighbor[node]:
                                neighbor.append(edge[0])
            three_hops_neighbor[node] = neighbor
        return three_hops_neighbor

    def get_edges_1_hop_neighbour(self):
        """获得一张图所有边的一阶邻居边，并以字典形式返回"""
        graph = nx.to_undirected(self.graph)
        edges_list = nx.edges(graph)
        edges_one_hop_neighbor = {}
        for edge in edges_list:
            neighbor_item = []
            for other_edge in edges_list:
                if edge != other_edge:
                    if edge[0] in other_edge:
                        neighbor_item.append(other_edge)
                    elif edge[1] in other_edge:
                        neighbor_item.append(other_edge)
            edges_one_hop_neighbor[edge] = neighbor_item
        return edges_one_hop_neighbor

    def get_nodes_degree_feature(self, neighbors_list):
        nodes_list = nx.nodes(self.graph)
        nodes_feature = {}
        for node in nodes_list:
            feature_item = []
            for neighbor in neighbors_list[node]:
                feature_item.append(nx.degree(self.graph, neighbor))
            nodes_feature[node] = feature_item
        return nodes_feature

    def get_nodes_1_hop_neighbor_degree_feature(self):
        node_one_hop_neighbor = self.get_nodes_1_hop_neighbors()
        return self.get_nodes_degree_feature(node_one_hop_neighbor)

    def get_nodes_2_hops_neighbor_degree_feature(self):
        node_two_hop_neighbor = self.get_nodes_2_hops_neighbors()
        return self.get_nodes_degree_feature(node_two_hop_neighbor)

    def get_nodes_3_hops_neighbor_degree_feature(self):
        node_three_hop_neighbor = self.get_nodes_3_hops_neighbor()
        return self.get_nodes_degree_feature(node_three_hop_neighbor)


class RelationalOperator:
    """操作算子集合，relational operator"""
    def op_Hadamard(self, x):
        return np.dot(x, x.T)

    def op_mean(self, x):
        return np.mean(x)

    def op_sum(self, x):
        return np.sum(x)

    def op_maximum(self, x):
        return np.maximum(x)
    #
    # def op_WeightLp(self, xi, x, p):
    #     return np.sum(np.power(np.abs(xi - x), p))
    #
    # def op_RBF(self, xi, x, thea):
    #     return np.exp(-1 / thea**2 * np.sum(np.power((xi - x), 2)))


# def update_nodes_feature_map(graph, x, t):
#     """更新节点与特征的映射关系
#     注意：t是上一层的层数，从0开始
#     """
#     update_feature_map = {}
#     nodes_list = graph.nodes()
#     for node in nodes_list:
#         update_feature_map[node] = x[node, t]
#     return update_feature_map
#
#
# def generate_nodes_new_feature(graph, nodes_neighbor, update_feature_map):
#     """在更新节点与特征的映射关系后，生成由上一特征层得到的新特征值，用对应节点对应的新的特征值"""
#     graph = nx.to_undirected(graph)
#     nodes_list = graph.nodes()
#     nodes_feature = {}
#     for node in nodes_list:
#         feature_item = []
#         for neighbor in nodes_neighbor[node]:
#             feature_item.append(update_feature_map[neighbor])
#         nodes_feature[node] = feature_item
#     return nodes_feature


def generate_node_next_layer_feature(graph, nodes_neighbor, x, t, fea_op):
    """不断地生成新的特征层，直至到达最大特征层，或者没有新的特征生成停止"""

    def update_nodes_feature_map(graph, x, t):
        """更新节点与特征的映射关系
        注意：t是上一层的层数，从0开始
        """
        update_feature_map = {}
        nodes_list = graph.nodes()
        for node in nodes_list:
            update_feature_map[node] = x[node, t]
        return update_feature_map

    def generate_nodes_new_feature(graph, nodes_neighbor, update_feature_map):
        """在更新节点与特征的映射关系后，生成由上一特征层得到的新特征值，用对应节点对应的新的特征值"""
        graph = nx.to_undirected(graph)
        nodes_list = graph.nodes()
        nodes_feature = {}
        for node in nodes_list:
            feature_item = []
            for neighbor in nodes_neighbor[node]:
                feature_item.append(update_feature_map[neighbor])
            nodes_feature[node] = feature_item
        return nodes_feature

    update_feature_map = update_nodes_feature_map(graph, x, t - 1)
    nodes_list = graph.nodes
    update_nodes_feature = generate_nodes_new_feature(
        graph, nodes_neighbor, update_feature_map)
    for node in nodes_list:
        x[node, t] = fea_op(np.array(update_nodes_feature[node]))
    return x


def feature_diffusion(graph, X, thea, t):
    """特征扩散，X(t)=(1-thea)*LX(t-1)+thea*X"""
    L = np.array(GetGraphMessage(graph).L, dtype=np.float32)
    X_0 = np.array(X, dtype=np.float32)
    X_t_1 = X_0
    for i in range(1, t + 1):
        X_t = (1 - thea) * np.matmul(L, X_t_1) + thea * X_0
        X_t_1 = X_t
    X = np.concatenate([X_0, X_t_1], 1)
    return X


def main(graph, t):
    """t是特征层数，从0开始"""
    node_list = nx.nodes(graph)
    X = np.zeros(shape=(nx.number_of_nodes(graph), t))
    PageRank = nx.pagerank(graph, alpha=0.8)
    for node in node_list:
        # X[node, 0] = nx.degree(graph, node)
        # X[node, 0] = nx.triangles(graph, node)
        X[node, 0] = PageRank[node]

    op = RelationalOperator()

    graph_element = GraphElementFeature(graph)
    node_one_hop_neighbor = graph_element.get_nodes_1_hop_neighbors()
    # 操作算子的选择是关键，操作算子的不同嵌套会得到不同的效果
    # for layer_number in range(1, t):
    #     X = generate_node_next_layer_feature(
    #         graph, node_one_hop_neighbor, X, layer_number, op.op_mean)
    #     if (X[:, layer_number - 1] == X[:, layer_number]).all():
    #         break
    #     if (abs((np.array(X[:, layer_number - 1]) -
    #              np.array(X[:, layer_number]))) < 1e-1).all():
    #         break
    X = generate_node_next_layer_feature(graph, node_one_hop_neighbor, X, 1, op.op_mean)
    X = generate_node_next_layer_feature(graph,node_one_hop_neighbor,X,2,op.op_sum)

    # 将矩阵里的全零列向项去除
    for i in range(X.shape[1]-1,1,-1):
        if (X[:,i] == 0).all():
            last_non_col = i
    try:
        X = X[:, 0:last_non_col]
    except NameError:
        X = X

    return X


def main1(graph, t):
    """t是特征层数，从0开始"""
    node_list = nx.nodes(graph)
    X = np.zeros(shape=(nx.number_of_nodes(graph), t))
    PaperRank = nx.pagerank(graph,alpha=0.8)
    for node in node_list:
        X[node, 0] = nx.degree(graph, node)
        # X[node, 0] = nx.triangles(graph,node)
        # X[node, 0] = PaperRank[node]

    op = RelationalOperator()

    graph_element = GraphElementFeature(graph)
    node_one_hop_neighbor = graph_element.get_nodes_1_hop_neighbors()
    # 操作算子的选择是关键，操作算子的不同嵌套会得到不同的效果
    for layer_number in range(1, t):
        X = generate_node_next_layer_feature(
            graph, node_one_hop_neighbor, X, layer_number, op.op_mean)
        if (X[:, layer_number - 1] == X[:, layer_number]).all():
            break
        if (abs((np.array(X[:, layer_number - 1]) -
                 np.array(X[:, layer_number]))) < 1e-1).all():
            break

    # 将矩阵里的全零列向项去除
    for i in range(X.shape[1]-1,1,-1):
        if (X[:,i] == 0).all():
            last_non_col = i
    try:
        X = X[:, 0:last_non_col]
    except NameError:
        X = X

    return X

def main2(graph, t):
    """t是特征层数，从0开始"""
    node_list = nx.nodes(graph)
    X = np.zeros(shape=(nx.number_of_nodes(graph), t))
    PaperRank = nx.pagerank(graph,alpha=0.8)
    for node in node_list:
        # X[node, 0] = nx.degree(graph, node)
        # X[node, 0] = nx.triangles(graph,node)
        X[node, 0] = PaperRank[node]

    op = RelationalOperator()

    graph_element = GraphElementFeature(graph)
    node_one_hop_neighbor = graph_element.get_nodes_1_hop_neighbors()
    # 操作算子的选择是关键，操作算子的不同嵌套会得到不同的效果
    for layer_number in range(1, t):
        X = generate_node_next_layer_feature(
            graph, node_one_hop_neighbor, X, layer_number, op.op_mean)
        if (X[:, layer_number - 1] == X[:, layer_number]).all():
            break
        if (abs((np.array(X[:, layer_number - 1]) -
                 np.array(X[:, layer_number]))) < 1e-1).all():
            break

    # 将矩阵里的全零列向项去除
    for i in range(X.shape[1]-1,1,-1):
        if (X[:,i] == 0).all():
            last_non_col = i
    try:
        X = X[:, 0:last_non_col]
    except NameError:
        X = X

    return X

if __name__ == "__main__":
    """
    import pickle as pkl
    import os
    
    path = os.getcwd()

    graph = nx.karate_club_graph()
    X = main(graph, 3)
    # X = feature_diffusion(graph,X,0.5,34)
    Y = main1(graph,2)
    # Y = feature_diffusion(graph,Y,0.5,34)
    # X = np.concatenate((np.array(X[:,1]).reshape(34,1),np.array(Y[:,0]).reshape(34,1)),axis=1)
    Z = main2(graph,2)
    X = np.concatenate((X,Y),axis=1)
    X = np.concatenate((X,Z),axis=1)
    print(X)
    print(X.shape)

    import pickle as pkl
    import os
    print(os.getcwd())

    with open(path+"\X_DeepGL_karate.pkl", "wb") as f:
        pkl.dump(X, f)
        f.close()

    # with open("F:\Windows10\Desktop\X.pkl", "rb") as f:
    #     Y = pkl.load(f)
    #     print(Y)

    # graph = GraphProcessing("F:\Code\python\DeepGL\graph\\frb40-19-5.mtx",first_line=0).new_graph()
    # X = main(graph,nx.number_of_nodes(graph))
    # print(X)
    # print(X.shape)
    #
    #
    # with open(path+"\\"+"X_DeepGL_frb40_19_5.pkl", "wb") as f:
    #     u = pkl.dump(X,f)
    """

    graph_path = "F:\Windows10\Desktop\ENZYMES_g296\ENZYMES_g296.edges"
    graph = GraphProcessing(graph_path, first_line=0).new_graph()
    # 在ENZYMES_g296.edges文件中没有节点0，节点0孤立，需要新加节点0
    graph.add_node(0)
    X = np.nan_to_num(main(graph, 3))
    import pickle as pkl
    with open("F:\Windows10\Desktop\ENZYMES_g296\ENZYMES_g296_DeepGL_pagerank_X.pkl", "wb") as file:
        pkl.dump(X, file)
        file.close()

    def auc(X_path,X_label_path):
        import evaluate_with_AUC as auc
        import pickle as pkl
        with open(X_path, "rb") as file:
            X = pkl.load(file)
        X_label = np.array(np.loadtxt(X_label_path, dtype=int))[:, 1]
        auc.evaluate_with_LR(X, X_label)
        auc.evaluate_eith_SVM(X, X_label)

    X_path = "F:\Windows10\Desktop\ENZYMES_g296\ENZYMES_g296_DeepGL_pagerank_X.pkl"
    X_label_path = "F:\Windows10\Desktop\ENZYMES_g296\ENZYMES_g296.nodes"
    auc(X_path,X_label_path)

    X_node2vec_path = "F:\Windows10\Desktop\ENZYMES_g296\\reorder_ENZYMES_g296_node2vec_X.txt"
    auc(X_path,X_label_path)