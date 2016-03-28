import time
from numpy.linalg import *
import numpy as np
import pymysql
from sklearn.cluster import *
import warnings
import matplotlib.pyplot as plt


def get_coauthor():
    conn = pymysql.connect(host='127.0.0.1', user='root', passwd='123456', db='dblp_conf100', charset='utf8')
    cur = conn.cursor()
    author_count = cur.execute("SELECT * FROM author")
    paper_count = cur.execute("SELECT * FROM paper")
    author_paper = np.zeros((author_count, paper_count))
    cur.execute("SELECT * FROM author_paper")
    for r in cur:
        author_paper[r[0] - 1][r[2] - 1] = 1
    cur.close()
    conn.close()
    coauthor = author_paper.dot(author_paper.T)
    return coauthor


def jpc_julei(coauthor):
    author_count = coauthor.__len__()
    eigval, eigvet = eigh(coauthor)
    eigvet_used = np.zeros((author_count, domain))
    for i in range(author_count - domain, author_count):
        eigvet_used.T[i - author_count + domain] = eigvet.T[i]
    result = KMeans(n_clusters=domain)
    result.fit(eigvet_used)
    return result.labels_


def pujulei(coauthor):
    result = SpectralClustering(n_clusters=domain, affinity='precomputed')
    result.fit(coauthor)
    return result.labels_


def show_labels(labels):
    la = ""
    for lab in labels:
        if len(str(lab)) == 1:
            la += "  " + str(lab)
        else:
            la += " " + str(lab)
    print(la)


def show_clunum(labels):
    clunum = np.zeros(domain)
    for lab in labels:
        clunum[lab] = clunum[lab] + 1
    print(clunum)


def test():
    dimen = 2
    data_red = 1.5 * np.random.randn(500, dimen) + 10
    data_green = np.random.randn(500, dimen) + 10
    data_green.T[0] = data_green.T[0] + 4
    data_green.T[1] = data_green.T[1] + 4
    data_blue = np.random.randn(500, dimen) + 10
    data_blue.T[0] = data_blue.T[0] + 5
    data_blue.T[1] = data_blue.T[1] - 5
    data = np.append(np.append(data_green, data_red, axis=0), data_blue, axis=0)
    # kmeans聚类
    start = time.clock()
    result = KMeans(n_clusters=domain)
    result.fit(data)
    show_error_count(result.labels_)
    print("kmeans用时" + str(time.clock() - start) + "秒")
    # 转换相似度矩阵
    datalen = data.__len__()
    coauthor = np.zeros((datalen, datalen), dtype=np.int)
    for i in np.arange(0, datalen):
        for j in np.arange(0, datalen):
            sqrt = norm((data[j] - data[i]))
            if sqrt > 3 or i == j:
                coauthor[i][j] = 0
            else:
                if sqrt <= 3 and sqrt > 2:
                    coauthor[i][j] = 1
                else:
                    if sqrt <= 2 and sqrt > 1:
                        coauthor[i][j] = 2
                    else:
                        coauthor[i][j] = 3
        coauthor[i][i] = np.sum(coauthor[i])
    # jpc聚类效果
    start = time.clock()
    show_error_count(jpc_julei(coauthor))
    print("自创聚类用时" + str(time.clock() - start) + "秒")
    # 普聚类效果
    start = time.clock()
    show_error_count(pujulei(coauthor))
    print("谱聚类用时" + str(time.clock() - start) + "秒")
    print(data[0])
    # 图形化展示聚类效果
    # plt.plot(data_red.T[0], data_red.T[1], 'ro', data_green.T[0], data_green.T[1],
    #          'go', data_blue.T[0], data_blue.T[1], 'bo')
    # plt.show()


def show_error_count(labels):
    wrong_count = 0
    for i in range(0, 3):
        count_0 = 0
        count_1 = 0
        count_2 = 0
        for j in range(0 + i * 500, 500 + i * 500):
            if (labels[j] == 0):
                count_0 = count_0 + 1
            if (labels[j] == 1):
                count_1 = count_1 + 1
            if (labels[j] == 2):
                count_2 = count_2 + 1
        if max(count_2, count_1, count_0) == count_0:
            wrong_count += count_2 + count_1
        else:
            if max(count_2, count_1, count_0) == count_1:
                wrong_count += count_2 + count_0
            else:
                wrong_count += count_0 + count_1
    print("分类错误的个数：" + str(wrong_count))


start = time.clock()
domain = 3
warnings.filterwarnings("ignore")
coau = get_coauthor()
show_clunum(jpc_julei(coau))
show_clunum(pujulei(coau))
test()
print("总用时" + str(time.clock() - start) + "秒")
