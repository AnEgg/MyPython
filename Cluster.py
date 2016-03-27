import time
from numpy.linalg import *
import numpy as np
import pymysql
from sklearn.cluster import *
import scipy.stats as sta
import matplotlib.pyplot as plt


def get_coautho():
    conn = pymysql.connect(host='127.0.0.1', user='root', passwd='123456', db='dblp_conf100', charset='utf8')
    cur = conn.cursor()
    author_count = cur.execute("SELECT * FROM author")
    paper_count = cur.execute("SELECT * FROM paper")
    author_paper = np.mat(np.empty((author_count, paper_count)), dtype='int16')
    cur.execute("SELECT * FROM author_paper")
    for r in cur:
        author_paper.put((r[0] - 1) * paper_count + r[2] - 1, 1)
    cur.close()
    conn.close()
    coauthor = author_paper * author_paper.T
    return coauthor, author_count


def jpc_julei(coauthor, author_count):
    eigval, eigvet = eigh(coauthor)
    eigvet_used = np.mat(np.empty((domain, author_count)))
    for i in range(author_count - domain, author_count):
        eigvet_used[i - author_count + domain] = np.copy(eigvet.T[i])
    result = KMeans(n_clusters=domain)
    result.fit(eigvet_used.T)
    la = ""
    for lab in result.labels_:
        la += str(lab) + " "
    print(la)


def pujulei(coauthor):
    result = SpectralClustering(n_clusters=domain, affinity='precomputed')
    result.fit(coauthor)
    la = ""
    for lab in result.labels_:
        la += str(lab) + " "
    print(la)


domain = 50;
start = time.clock()
# au_pa = get_coautho()
# jpc_julei(au_pa[0], au_pa[1])
# pujulei(au_pa[0])
bins = np.arange(-4, 5, 0.1, dtype='float16')
print(bins)
b = sta.norm.pdf(bins)  # norm是正态分布
plt.plot(bins, b)
plt.show()
print(time.clock() - start)
