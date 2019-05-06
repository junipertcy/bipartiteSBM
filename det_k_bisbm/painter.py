""" plots """

import matplotlib.pyplot as plt
import scipy.sparse as sps
from .utils import *
from itertools import combinations

from clusim.clustering import Clustering
import clusim.sim as sim
from sklearn import manifold


def paint_block_mat_from_e_rs(e_rs, save2file=False, **kwargs):
    plt.figure(figsize=(3, 3))
    frame = plt.gca()

    size = []
    x_index = []
    y_index = []
    for i in range(len(e_rs)):
        for j in range(len(e_rs)):
            x_index.append(i + 0.5)
            y_index.append(j + 0.5)
            size.append(e_rs[i][j])

    plt.scatter(x_index,
                y_index,
                marker='s',
                color='k',
                alpha=0.8,
                facecolors='k',
                #             edgecolors='k',
                s=size / np.max(size) * 100,
                label='')

    plt.ylabel('')
    plt.xlabel('')

    # and a legend
    # plt.legend(loc='upper right')

    # set the figure boundaries
    plt.xlim([0 - 0.2, len(e_rs) + 0.2])
    plt.ylim([0 - 0.2, len(e_rs) + 0.2])

    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

    if save2file:
        try:
            path = kwargs["path"]
        except KeyError:
            raise (KeyError, "Please specify `path` for save2file.")
        plt.savefig(path + '.eps', format='eps', dpi=300)


def paint_block_mat(mb, edgelist, save2file=False, **kwargs):
    mb = np.asanyarray(mb, dtype=int)
    e_rs, _ = assemble_e_rs_from_mb(edgelist, mb)

    plt.figure(figsize=(3, 3))
    frame = plt.gca()

    size = []
    x_index = []
    y_index = []
    for i in range(len(e_rs)):
        for j in range(len(e_rs)):
            x_index.append(i + 0.5)
            y_index.append(j + 0.5)
            size.append(e_rs[i][j])

    plt.scatter(x_index,
                y_index,
                marker='s',
                color='k',
                alpha=0.8,
                facecolors='k',
                #             edgecolors='k',
                s=size / np.max(size) * 100,
                label='')

    plt.ylabel('')
    plt.xlabel('')

    # and a legend
    # plt.legend(loc='upper right')

    # set the figure boundaries
    plt.xlim([0 - 0.2, len(e_rs) + 0.2])
    plt.ylim([0 - 0.2, len(e_rs) + 0.2])

    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

    if save2file:
        try:
            path = kwargs["path"]
        except KeyError:
            raise (KeyError, "Please specify `path` for save2file.")
        plt.savefig(path + '.eps', format='eps', dpi=300)


def paint_sorted_adj_mat(mb, edgelist, save2file=False, **kwargs):
    mb = np.argsort(mb)
    A = np.zeros([len(mb), len(mb)])
    for edge in edgelist:
        e0 = int(edge[0])
        e1 = int(edge[1])
        A[np.argwhere(mb == e0)[0][0]][np.argwhere(mb == e1)[0][0]] += 1
        A[np.argwhere(mb == e1)[0][0]][np.argwhere(mb == e0)[0][0]] += 1
    M = sps.csr_matrix(A)
    plt.spy(M, markersize=0.25, marker=".")
    if save2file:
        try:
            path = kwargs["path"]
        except KeyError:
            raise (KeyError, "Please specify `path` for save2file.")
        plt.savefig(path + '.eps', format='eps', dpi=300)


def paint_trace(oks, save2file=False, **kwargs):
    from matplotlib import collections as mc

    trace = list(oks.trace_mb.keys())
    lines = []
    for ind, i in enumerate(trace):
        if ind != len(trace) - 1:
            lines += [(trace[ind], trace[ind + 1])]

    # lines = lines[20:]
    lc = mc.LineCollection(lines, linewidths=2, color="#0074D9")
    fig, ax = plt.subplots(figsize=(3, 3), dpi=300)
    ax.add_collection(lc)

    # Pink circle marks the optimal point (ka, kb)
    summary = oks.summary()
    ax.scatter(summary["ka"], summary["kb"], marker="o", c="pink", s=200, alpha=0.5)

    # Black numbers indicate ordered points where graph partition takes place
    for idx, point in enumerate(list(oks.bookkeeping_dl.keys())):
        #     plt.scatter(point[0], point[1], marker='${}$'.format(idx), c="black", edgecolors="none", s=100)
        plt.scatter(point[0], point[1], marker='x', c="#FF4136", edgecolors="none", s=20)

    # for idx, point in enumerate(list(oks.confident_desc_len.keys())):
    #     plt.scatter(point[0], point[1], marker='${}$'.format(idx), c="black", edgecolors="none", s=100)
    #     ax.scatter(point[0], point[1], marker='o', color="#0074D9", s=5)

    ax.autoscale()
    ax.margins(0.1)

    ax.set_aspect(1)
    plt.xlabel("number of type-a communities: ka")
    plt.ylabel("number of type-b communities: kb")
    plt.xticks(np.arange(0, 61, 10))
    plt.yticks(np.arange(0, 61, 10))

    plt.axvline(4)
    plt.axhline(6)

    # ax.set_xlim([2, 8])
    # ax.set_ylim([2, 8])
    # plt.xticks(np.arange(2, 9, 1))
    # plt.yticks(np.arange(2, 9, 1))

    # plt.show()
    if save2file:
        try:
            path = kwargs["path"]
        except KeyError:
            raise (KeyError, "Please specify `path` for save2file.")
        plt.savefig(path + '.eps', format='eps', dpi=300)


def paint_dl_trace(oks, save2file=False, **kwargs):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    desc_len_list = []
    na = oks.summary()["na"]
    nb = oks.summary()["nb"]
    e = oks.summary()["e"]

    for idx, key in enumerate(oks.trace_mb.keys()):
        mb = oks.trace_mb[key]
        nr = assemble_n_r_from_mb(mb)
        desc_len_list += [get_desc_len_from_data(na, nb, e, key[0], key[1], oks.edgelist, mb, nr=nr)]
    ax.autoscale()
    ax.margins(0.1)

    plt.xlabel("steps")
    plt.ylabel("description length")
    if save2file:
        try:
            path = kwargs["path"]
        except KeyError:
            raise (KeyError, "Please specify `path` for save2file.")
        plt.savefig(path + '.eps', format='eps', dpi=300)
    plt.plot(desc_len_list, 'o-')


def paint_similarity_trace(b, oks, save2file=False, **kwargs):

    clu_base = Clustering()
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    e_sim_list = []
    b = clu_base.from_membership_list(b)
    for g in oks.trace_mb.values():
        clu = Clustering()
        clu.from_membership_list(list(g))
        e_sim_list += [sim.element_sim(clu_base, clu)]

    ax.autoscale()
    ax.margins(0.1)
    # ax.set_aspect(1)
    plt.xlabel("steps")
    plt.ylabel("Element-centric similarity")
    plt.yticks(np.linspace(0, 1, 5))

    if save2file:
        try:
            path = kwargs["path"]
        except KeyError:
            raise (KeyError, "Please specify `path` for save2file.")
        plt.savefig(path + '.eps', format='eps', dpi=300)
    plt.plot(e_sim_list)


def paint_mds(oks):
    l2 = len(oks.trace_mb.keys())
    l = int(l2 ** 0.5)
    X = np.zeros([l2, l2])
    for idx_1, pair_1 in enumerate(combinations(range(1, l + 1), 2)):
        b = oks.trace_mb[pair_1]
        clu_1 = Clustering()
        clu_1.from_membership_list(b)
        for idx_2, pair_2 in enumerate(combinations(range(1, l + 1), 2)):
            b = oks.trace_mb[pair_2]
            clu_2 = Clustering()
            clu_2.from_membership_list(b)

            X[idx_1][idx_2] = 1 - sim.element_sim(clu_1, clu_2)
            X[idx_2][idx_1] = X[idx_1][idx_2]

    def _plot_embedding(X, title=None):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure(figsize=(20, 20))
        for ind, i in enumerate(range(X.shape[0])):
            plt.text(X[i, 0], X[i, 1], str(list(oks.trace_mb.keys())[ind]), color=plt.cm.Set1(1 / 10.),
                     fontdict={'weight': 'bold', 'size': 12})
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

    clf = manifold.MDS(n_components=2, n_init=10, max_iter=10000, dissimilarity="precomputed")
    X_mds = clf.fit_transform(X)
    _plot_embedding(X_mds)
