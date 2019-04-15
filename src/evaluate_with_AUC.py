# coding = utf-8
# python 3.6.7
# Created by wuyang at2019/3/10
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
# from keras import models
# import keras
import tensorflow as tf
import scipy
# import scipy.sparse
import scipy.spatial.distance
import scipy.ndimage.fourier as snf

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn import metrics
from sklearn import svm
from threading import Thread
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import example_testing


def reorder_node_feature_matrix(file_order):
    """将生成的节点特征矩阵，按照节点序号递增重新排列，并返回重新排列好的特征矩阵，
    array的序号是从0开始，即第1行就是节点序号为1的特征表示"""
    X = np.loadtxt("F:\Windows10\Desktop\ENZYMES_g" + file_order + "\ENZYMES_g" + file_order + "_node.txt")
    reorder_X = np.zeros(shape=(X.shape[0], X.shape[1] - 1))

    # 将读取特征矩阵中第一列的节点序号提取出来，注意在原文件中节点序号从1开始
    node_serial_set = np.array(X[:, 0], dtype=int)

    for i in range(len(node_serial_set)):
        # 使节点序号和array序号匹配
        reorder_X[node_serial_set[i] - 1, :] = np.array(X[i, :])[1:]

    return reorder_X


def evaluate_with_LR(X, X_label):
    """使用LR来做节点分类，并用AUC评估"""
    sample_train, sample_test, label_train, label_test = train_test_split(X, X_label, test_size=0.3,
                                                                          random_state=30)

    classifier = LogisticRegression()
    classifier.fit(sample_train, label_train)
    pred = classifier.predict(sample_test)

    fpr, tpr, thresholds = metrics.roc_curve(label_test, pred, pos_label=2)
    LR_AUC = metrics.auc(fpr, tpr)
    LR_SCORE = classifier.score(sample_test, label_test)
    print("LR_AUC: ", LR_AUC)
    print("LR_SCORE: ", LR_SCORE)


def evaluate_eith_SVM(X, X_label):
    """使用SVC来做节点分类，并用AUC评估"""
    sample_train, sample_test, label_train, label_test = train_test_split(X, X_label, test_size=0.3,
                                                                          random_state=30)
    # kernel = "rbf", "poly", "linear", "sigmoid", "precomputed"
    classifier = SVC(kernel="linear", degree=2, gamma=1, coef0=0)
    # classifier = SVC()
    classifier.fit(sample_train, label_train)

    pred = classifier.predict(sample_test)

    fpr, tpr, thresholds = metrics.roc_curve(label_test, pred, pos_label=2)
    SVM_AUC = metrics.auc(fpr, tpr)
    SVM_SCORE = classifier.score(sample_test, label_test)
    print("SVM_AUC: ", SVM_AUC)
    print("SVM_SCORE: ", SVM_SCORE)
    pass



if __name__ == "__main__":

    def source_file_index_begin_with_zero(source_path=str,object_path=str):
        """将源数据里的节点序号初始化为从0开始，需要将文件前两行数据先删除，再处理"""
        """配套PGD使用"""
        x = np.loadtxt(source_path)
        if np.min(x) != 0:
            for i in range(x.shape[0]):
                x[i, 0] -= 1
                x[i, 1] -= 1

            with open(object_path, "w") as f:
                for i in range(x.shape[0]):
                    f.write(str(int(x[i, 0])))
                    f.write(" ")
                    f.write(str(int(x[i, 1])))
                    f.write("\n")
                f.close()
    # source_file_index_begin_with_zero("F:\Windows10\Desktop\\frb40-19-5.mtx","F:\Windows10\Desktop\\frb40-19-5.mtx")

    # y = np.array([2, 2, 1, 2, 2, 2, 1, 1, 2])
    # y = np.array([1,1,0,1,1,1,0,0,1])
    # pred = np.array([0.09, 0.08, 0.07, 0.06, 0.055, 0.054, 0.053, 0.052, 0.051])
    # # fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
    # fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    # print("fpr: ", fpr, len(fpr))
    # print("tpr: ", tpr, len(tpr))
    # print("thresholds: ", thresholds, len(thresholds))
    # print("auc: ", metrics.auc(fpr, tpr))
    # plt.figure()
    # plt.plot(fpr,tpr,c="red")
    # # plt.show()

    breast_canaer = load_breast_cancer()
    sample = breast_canaer.data
    label = breast_canaer.target
    # print(sample.shape)
    # print(label.shape)

    sample_train, sample_test, label_train, label_test = train_test_split(sample, label, test_size=0.3, random_state=30)

    classifier = LogisticRegression()
    classifier.fit(sample_train, label_train)

    pred = classifier.predict(sample_test)
    fpr, tpr, thresholds = metrics.roc_curve(label_test, pred)
    LR_AUC = metrics.auc(fpr, tpr)
    print("LR_AUC: ", LR_AUC)

    LR_SCORE = classifier.score(sample_test, label_test)
    print("LR_SCORE: ", LR_SCORE)

    classifier = SVC(kernel="linear",degree=2,coef0=0,gamma=1)
    # classifier = SVC()
    classifier.fit(sample_train,label_train)

    pred = classifier.predict(sample_test)
    print(pred)
    print(label_test)
    fpr, tpr, thresholds = metrics.roc_curve(label_test, pred)
    SVM_AUC = metrics.auc(fpr, tpr)
    print("SVM_AUC: ", SVM_AUC)

    SVM_SCORE = classifier.score(sample_test, label_test)
    print("SVM_SCORE: ", SVM_SCORE)


    # def reorder_node_feature_matrix(file_order):
    #     """将生成的节点特征矩阵，按照节点序号递增重新排列，并返回重新排列好的特征矩阵，
    #     array的序号是从0开始，即第1行就是节点序号为1的特征表示"""
    #     X = np.loadtxt("F:\Windows10\Desktop\ENZYMES_g" + file_order + "\ENZYMES_g" + file_order + "_node.txt")
    #     reorder_X = np.zeros(shape=(X.shape[0], X.shape[1] - 1))
    #
    #     # 将读取特征矩阵中第一列的节点序号提取出来，注意在原文件中节点序号从1开始
    #     node_serial_set = np.array(X[:, 0], dtype=int)
    #
    #     for i in range(len(node_serial_set)):
    #         # 使节点序号和array序号匹配
    #         reorder_X[node_serial_set[i] - 1, :] = np.array(X[i, :])[1:]
    #
    #     return reorder_X
    #
    #
    # def evaluate_with_LR(X, X_label):
    #     """使用LR来做节点分类，并用AUC评估"""
    #     sample_train, sample_test, label_train, label_test = train_test_split(X, X_label, test_size=0.3,
    #                                                                           random_state=30)
    #
    #     classifier = LogisticRegression()
    #     classifier.fit(sample_train, label_train)
    #     pred = classifier.predict(sample_test)
    #
    #     fpr, tpr, thresholds = metrics.roc_curve(label_test, pred, pos_label=2)
    #     LR_AUC = metrics.auc(fpr, tpr)
    #     LR_SCORE = classifier.score(sample_test, label_test)
    #     print("LR_AUC: ", LR_AUC)
    #     print("LR_SCORE: ", LR_SCORE)
    #
    # def evaluate_eith_SVM(X, X_label):
    #     """使用SVC来做节点分类，并用AUC评估"""
    #     sample_train, sample_test, label_train, label_test = train_test_split(X, X_label, test_size=0.3,
    #                                                                           random_state=30)
    #     # kernel = "rbf", "poly", "linear", "sigmoid", "precomputed"
    #     classifier = SVC(kernel="linear", degree=2, gamma=1, coef0=0)
    #     # classifier = SVC()
    #     classifier.fit(sample_train, label_train)
    #
    #     pred = classifier.predict(sample_test)
    #
    #     fpr, tpr, thresholds = metrics.roc_curve(label_test, pred, pos_label=2)
    #     SVM_AUC = metrics.auc(fpr, tpr)
    #     SVM_SCORE = classifier.score(sample_test, label_test)
    #     print("SVM_AUC: ", SVM_AUC)
    #     print("SVM_SCORE: ", SVM_SCORE)
    #     pass

    file_order = str(296)

    X_node2vec_ENZYMES_g = reorder_node_feature_matrix(file_order)
    X_ENZYMES_g_label = np.array(np.loadtxt("F:\Windows10\Desktop\ENZYMES_g"+file_order+"\ENZYMES_g"+file_order+".nodes"), dtype=int)[:, 1]

    evaluate_with_LR(X_node2vec_ENZYMES_g, X_ENZYMES_g_label)
    evaluate_eith_SVM(X_node2vec_ENZYMES_g, X_ENZYMES_g_label)

