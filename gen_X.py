# 生成特征矩阵

from DeepGL.src import DeepGL
from DeepGL.src import node2vec_representation
import numpy as np
import pickle as pkl
import os
graph_id = "295"

graph_path = "F:\Code\python\DeepGL\graph\ENZYMES_g"+graph_id+"\ENZYMES_g"+graph_id+".edges"
graph = DeepGL.GraphProcessing(graph_path, first_line=0).new_graph()
# 在ENZYMES_g296.edges文件中没有节点0，节点0孤立，需要新加节点0
graph.add_node(0)

# 使用DeepGL生成特征矩阵X
X = np.nan_to_num(DeepGL.main(graph, 3))
gen_X_path = "F:\Code\python\DeepGL\emb\ENZYMES_g"+graph_id+"\ENZYMES_g"+graph_id
with open(gen_X_path+"_DeepGL_pagerank_X.pkl", "wb") as file:
    pkl.dump(X, file)
    file.close()


# 使用node2vec生成特征矩阵X
node2vec_node_X_path = gen_X_path+"_node2vec_node_X.txt"
if not os.path.exists(node2vec_node_X_path):
    node2vec_representation.node2vec_X(graph,node2vec_node_X_path)

def reorder_node_feature_matrix(reorder_file_path):
    """将生成的节点特征矩阵，按照节点序号递增重新排列，并返回重新排列好的特征矩阵，
    array的序号是从0开始，即第1行就是节点序号为1的特征表示,
    需要手动将第一行信息删除"""
    X = np.loadtxt(reorder_file_path)
    reorder_X = np.zeros(shape=(X.shape[0], X.shape[1]-1))

    # 将读取特征矩阵中第一列的节点序号提取出来，节点序号从0开始
    node_serial_set = np.array(X[:, 0], dtype=int)
    print(sorted(node_serial_set))
    for i in range(len(node_serial_set)):
        reorder_X[node_serial_set[i], :] = np.array(X[i, :])[1:]
    return reorder_X


# 将使用node2vec生成的TXT文档转为PKL文档
if not os.path.exists(gen_X_path+"_node2vec_node_X.pkl"):
    with open(gen_X_path+"_node2vec_node_X.pkl", "wb") as file:
        pkl.dump(reorder_node_feature_matrix(gen_X_path+"_node2vec_node_X.txt"), file)
        file.close()
# os.remove(gen_X_path+"_node2vec_node_X.txt")
pkl_size = os.path.getsize(gen_X_path+"_node2vec_node_X.pkl")
txt_size = os.path.getsize(gen_X_path+"_node2vec_node_X.txt")
print(pkl_size, txt_size)

if pkl_size == 0:
    os.remove(gen_X_path+"_node2vec_node_X.pkl")
