""" plots """
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.sparse as sps
from .utils import *
from itertools import combinations

from clusim.clustering import Clustering
import clusim.sim as sim
from sklearn import manifold
import numpy as np


def paint_block_mat_from_e_rs(e_rs, output=None, figsize=(3, 3), dpi=200, **kwargs):
    plt.figure(figsize=figsize)
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

    if output is not None:
        plt.savefig(output, dpi=dpi, transparent=True)


def paint_block_mat(mb, edgelist, output=None, figsize=(3, 3), dpi=200, **kwargs):
    mb = np.asanyarray(mb, dtype=int)
    e_rs = assemble_e_rs_from_mb(edgelist, mb)

    plt.figure(figsize=figsize)
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

    if output is not None:
        plt.savefig(output, dpi=dpi, transparent=True)


def paint_sorted_adj_mat(mb, edgelist, output=None, figsize=(10, 10), dpi=300, invert=True):
    font = {'family': 'serif'}
    plt.figure(figsize=(10, 10))
    fig, ax = plt.subplots()
    mb = np.argsort(mb)
    A = np.zeros([len(mb), len(mb)])
    for edge in edgelist:
        e0 = int(edge[0])
        e1 = int(edge[1])
        A[np.argwhere(mb == e0)[0][0]][np.argwhere(mb == e1)[0][0]] += 1
        A[np.argwhere(mb == e1)[0][0]][np.argwhere(mb == e0)[0][0]] += 1
    M = sps.csr_matrix(A)
    plt.spy(M, markersize=0.01, marker=",")
    plt.xlabel(f"(Node index $i$) / {len(mb)}", fontdict=font)
    plt.ylabel(f"(Node index $i$) / {len(mb)}", fontdict=font)

    plt.xticks(np.linspace(0, 1, 5) * len(mb), ('0', '0.25', '0.5', '0.75', '1'))
    plt.yticks(np.linspace(0, 1, 5) * len(mb), ('0', '0.25', '0.5', '0.75', '1'))
    if invert:
        plt.gca().invert_yaxis()
        ax.tick_params(axis="y", direction="in")
        ax.tick_params(axis="x", direction="in")
        ax.xaxis.set_ticks_position("bottom")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    if output is not None:
        plt.savefig(output, dpi=dpi, transparent=True)


def paint_trace(oks, output=None, figsize=(4, 4), dpi=200):
    from matplotlib.collections import LineCollection
    summary = oks.summary()
    trace = [(i[1], i[2]) for i in oks.trace_k]

    lines = []
    for ind, i in enumerate(trace):
        if ind != len(trace) - 1:
            lines += [(trace[ind], trace[ind + 1])]

    lines.pop(0)  # remove the first line segment to make it prettier

    lc = LineCollection(lines, linewidths=0.5, color="#0074D9", )
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.add_collection(lc)
    ax.autoscale()

    ax.tick_params(direction="in")

    x = [i[0] for j in lines for i in j]
    y = [i[1] for j in lines for i in j]

    ax.scatter(x, y)

    # Locate the mdl point (Pink circle marks the optimal point (ka, kb))
    ka = summary["ka"]
    kb = summary["kb"]
    plt.axvline(ka, color="#DDDDDD", linewidth=0.5)
    plt.axhline(kb, color="#DDDDDD", linewidth=0.5)
    #     ax.scatter(ka, kb, marker="o", c="pink", s=200, alpha=0.5)

    # Black numbers indicate ordered points where graph partition takes place
    for idx, point in enumerate(list(oks.bookkeeping_dl.keys())):
        plt.scatter(point[0], point[1], marker='x', c="#FF4136", edgecolors="none", s=20)

    ax.margins(0.01)

    ax.set_aspect(1)
    plt.xlabel("$K_a$")
    plt.ylabel("$K_b$")

    k = np.array(trace)
    k.flatten()
    k = np.max(k)
    plt.xticks(np.arange(0, k + 1, 2))
    plt.yticks(np.arange(0, k + 1, 2))
    plt.xlim([0, k + 1])
    plt.ylim([0, k + 1])

    # plt.show()
    if output is not None:
        plt.savefig(output, dpi=dpi, transparent=True)


def paint_dl_trace(oks, output=None, figsize=(4, 2), dpi=200, **kwargs):
    qc = oks.get__q_cache()
    na = oks.summary()["na"]
    nb = oks.summary()["nb"]
    e = oks.summary()["e"]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    desc_len_list = []

    for idx, key in enumerate(oks.trace_mb.keys()):
        mb = oks.trace_mb[key][1]
        nr = assemble_n_r_from_mb(mb)
        desc_len_list += [get_desc_len_from_data(na, nb, e, key[0], key[1], oks.edgelist, mb, nr=nr, q_cache=qc)]
    ax.autoscale()
    ax.margins(0.1)
    ax.tick_params(direction="in")

    plt.xlabel("steps")
    plt.ylabel("DL")
    plt.plot(desc_len_list, 'o-')
    if output is not None:
        plt.savefig(output, dpi=dpi, transparent=True)


def paint_similarity_trace(b, oks, output=None, figsize=(3, 3), dpi=200, **kwargs):
    clu_base = Clustering()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    e_sim_list = []
    clu_base.from_membership_list(b)
    for g in oks.trace_mb.values():
        clu = Clustering()
        clu.from_membership_list(g[1])
        e_sim_list += [sim.element_sim(clu_base, clu)]

    ax.autoscale()
    ax.margins(0.1)
    # ax.set_aspect(1)
    plt.xlabel("steps")
    plt.ylabel("Element-centric similarity")
    plt.yticks(np.linspace(0, 1, 5))
    ax.tick_params(direction="in")
    plt.plot(e_sim_list)
    if output is not None:
        plt.savefig(output, dpi=dpi, transparent=True)


def paint_landscape(oks, max_ka, max_kb, output=None, dpi=200,):
    mat = np.zeros([max_ka, max_kb])
    for i in oks.bookkeeping_dl.keys():
        try:
            mat[i[0] - 1, i[1] - 1] = oks.bookkeeping_dl[i]
        except IndexError:
            pass

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))  # setup the plot

    colors_undersea = plt.cm.terrain(np.linspace(0, 0.95, 256))
    colors_land = plt.cm.terrain(np.linspace(0.95, 1, 256))
    all_colors = np.vstack((colors_undersea, colors_land))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('terrain_map', all_colors)

    # define the bins and normalize
    bounds = np.geomspace(min(mat.flatten()), max(mat.flatten()), 256)
    norm = mpl.colors.BoundaryNorm(bounds, 256)

    ims = ax.imshow(mat, norm=norm, cmap=cmap, origin='lower', extent=[1, max_ka, 1, max_ka], rasterized=True)
    plt.xlabel("$K_a$")
    plt.ylabel("$K_b$")

    # scaled colorbar that aligns with the frame
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(ims, cax=cax, label="DL (unit: nat)", shrink=1)
    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")
    ax.xaxis.set_ticks_position("bottom")

    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)

    ax.set_xticks(np.arange(1.5, max_ka + 1, 4))
    ax.set_yticks(np.arange(1.5, max_kb + 1, 4))
    ax.set_xticklabels(np.arange(1, max_ka + 1, 4))
    ax.set_yticklabels(np.arange(1, max_kb + 1, 4))
    if output is not None:
        plt.savefig(output, dpi=dpi, transparent=True)


def paint_mds(oks, figsize=(20, 20)):
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

        plt.figure(figsize=figsize)
        for ind, i in enumerate(range(X.shape[0])):
            plt.text(X[i, 0], X[i, 1], str(list(oks.trace_mb.keys())[ind]), color=plt.cm.Set1(1 / 10.),
                     fontdict={'weight': 'bold', 'size': 12})
        plt.xticks([]), plt.yticks([])
        if title is not None:
            plt.title(title)

    clf = manifold.MDS(n_components=2, n_init=10, max_iter=10000, dissimilarity="precomputed")
    X_mds = clf.fit_transform(X)
    _plot_embedding(X_mds)
