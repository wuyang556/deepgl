# coding = utf-8 
# python 3.6.7 
# Created by wuyang at2019/4/15

from DeepGL.src import evaluate_with_AUC
import pickle as pkl
import numpy as np


graph_id = "295"


def X_path(feature_name_id, graph_id):
    features_name = ["degree", "triangles", "pagerank"]
    path = "F:\Code\python\DeepGL\emb\ENZYMES_g" + graph_id + "\\" + "ENZYMES_g" + graph_id + "_DeepGL_" + \
           features_name[feature_name_id] + "_X.pkl"
    return open(path, "rb")
# 使用DeepGL生成特征矩阵做节点分类实验
X_degree = pkl.load(X_path(0,graph_id))
X_triangles = pkl.load(X_path(1, graph_id))
X_pagerank = pkl.load(X_path(2, graph_id))
X_DeepGL = np.concatenate((X_degree,X_triangles,X_pagerank), axis=1)

X_label_path = "F:\Code\python\DeepGL\graph\ENZYMES_g"+graph_id+"\ENZYMES_g"+graph_id+".nodes"
X_label = np.array(np.loadtxt(X_label_path), dtype=int)[:, 1]
print("DeepGL:")
evaluate_with_AUC.evaluate_with_LR(X_DeepGL, X_label)
evaluate_with_AUC.evaluate_eith_SVM(X_DeepGL, X_label)

X_node2vec_path = "F:\Code\python\DeepGL\emb\ENZYMES_g"+graph_id+"\ENZYMES_g"+graph_id+"_node2vec_node_X.pkl"
# 使用node2vec生成特征矩阵来做节点分类实验
with open(X_node2vec_path, "rb") as file:
    X_node2vec = pkl.load(file)
print("node2vec:")
evaluate_with_AUC.evaluate_with_LR(X_node2vec, X_label)
evaluate_with_AUC.evaluate_eith_SVM(X_node2vec, X_label)

