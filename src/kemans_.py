# coding = utf-8 
# python 3.6.7 
# Created by wuyang at2019/3/26

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn import metrics
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import os
import networkx as nx
import deepgl
import node2vec_embeding

graph_frb40 = deepgl.GraphProcessing(path="F:\Code\python\DeepGL\graph\\frb40-19-5.mtx", graph_direct=0, first_line=0).new_graph()
A_frb40 = np.array(nx.linalg.adjacency_matrix(graph_frb40).todense())

# graph = DeepGL.GraphProcessing(path="F:\Code\python\DeepGL\graph\\frb40-19-5.mtx",graph_direct=0,first_line=0).new_graph()
path = os.getcwd()

graph = nx.karate_club_graph()
A = np.array(nx.linalg.adjacency_matrix(graph).todense())
PagerRank = nx.pagerank(graph)
# print(PagerRank)
# pos = nx.kamada_kawai_layout(graph)
# nx.draw_networkx(graph,pos=pos)
# plt.show()
# print(nx.k_core(graph))

X = np.zeros(shape=(34,3))
for i in range(nx.number_of_nodes(graph)):
    X[i, 0] = PagerRank[i]
    X[i, 1] = nx.degree(graph, i)
    X[i, 2] = nx.triangles(graph, i)
# print(X)

with open(path+"\X_DeepGL_karate.pkl","rb") as f:
    X_DeepGL_karate = pkl.load(f)
    f.close()
# print(X_DeepGL_karate.shape)


with open("F:\Code\python\\X_node2vec.pkl", "rb") as f:
    X_node2vec_karate = pkl.load(f)
    f.close()

with open(path+"\\"+"X_DeepGL_frb40_19_5.pkl","rb") as f:
    X_DeepGL_frb40_19_5 = pkl.load(f)


# 使用K-means聚类
def K_means(X,n_clusters=int,random_state=int):
    kmeans = KMeans(n_clusters=n_clusters,random_state=random_state).fit_predict(X)
    # print(KMeans(n_clusters=n_clusters,random_state=random_state).fit(X).cluster_centers_)
    plt.figure(figsize=(8, 6))
    # plt.scatter(X[:, 0], X[:, 1], c=kmeans)
    # plt.show()
    print("DeepGL_DB_value:",metrics.davies_bouldin_score(X, kmeans))
    for i in range(0,n_clusters):
        print(i,list(kmeans).count(i))

    index_0 = []
    index_1 = []
    index_dict = {}
    for i in range(len(list(kmeans))):
        if list(kmeans)[i] == 0:
            index_0.append(i)
        elif list(kmeans)[i] == 1:
            index_1.append(i)
    index_dict[0] = index_0
    index_dict[1] = index_1
    print(index_dict)


# 使用谱聚类
def Spectral_cluster(X,n_clusters=int, gamma=int):
    spectral = SpectralClustering(n_clusters=n_clusters, gamma=gamma).fit_predict(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=spectral)
    # plt.show()
    for i in range(0, n_clusters):
        print(i, list(spectral).count(i))

if __name__ == "__main__":

    graph_karate = nx.karate_club_graph()
    A_karate = np.array(nx.linalg.adjacency_matrix(graph).todense())
    D_karate = np.zeros(shape=(34,3))
    # print(D_karate)
    # K_means(A_karate, n_clusters=2,random_state=10)
    PagerRank = nx.pagerank(graph_karate)
    for node in range(nx.number_of_nodes(graph_karate)):
        # D_karate[node, 0] = nx.degree(graph_karate, node)
        # D_karate[node, 1] = nx.triangles(graph_karate, node)
        # D_karate[node, 2] = PagerRank[node]
        D_karate[node, 0] = PagerRank[node]
        D_karate[node, 1] = nx.triangles(graph_karate, node)
        D_karate[node, 2] = nx.degree(graph_karate, node)

    K_means(D_karate,n_clusters=2,random_state=10)

    # node2vec_embeding.node2vec_embedding(graph_karate,dimensions=3)

    with open("F:\Code\python\DeepGL\src\X_node2vec_karate_d1.pkl","rb") as f:
        X_node2vec_karate_d1 = pkl.load(f)

    # K_means(np.array(X_node2vec_karate_d1).reshape(34,1),n_clusters=2,random_state=10)

    with open("F:\Code\python\DeepGL\src\X_node2vec_karate_d2.pkl","rb") as f:
        X_node2vec_karate_d2 = pkl.load(f)

    # K_means(np.array(X_node2vec_karate_d2).reshape(34,2),n_clusters=2,random_state=10)

    with open("F:\Code\python\DeepGL\src\X_node2vec_karate_d3.pkl","rb") as f:
        X_node2vec_karate_d3 = pkl.load(f)
    # K_means(np.array(X_node2vec_karate_d3).reshape(34,3),n_clusters=2,random_state=10)

