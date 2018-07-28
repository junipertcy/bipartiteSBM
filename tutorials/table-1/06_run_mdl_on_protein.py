# 06_run_bimdl_on_protein.py

import os
import sys
module_path = os.path.abspath(os.path.join('../../'))
if module_path not in sys.path:
    sys.path.append(module_path)


filename = "../../dataset/empirical/protein-complex_739-drug_680.edgelist"
na = 739
nb = 680
avg_deg = 5.2

n = na + nb

k = int( (n * avg_deg / 2) ** 0.5)

import graph_tool.all as gt
from det_k_bisbm.utils import get_desc_len_from_data, get_desc_len_from_data_uni

edges = []
with open(filename, 'r') as f:
    for line in f:
        edges += [list(map(int, line.replace('\n', '').split(' ')))]

g = gt.Graph()
g.add_edge_list(edges)

g.set_directed(False)

dl_B_pair = []
counts = 0
for _ in range(100):
    state = gt.minimize_blockmodel_dl(g)
    B = state.B
    b = list(state.b)
    ka = set(b[:na])
    kb = set(b[na:])
    is_bipartite = True
    if len(ka.intersection(kb)) == 0: # bipartite
        counts += 1
        dl = get_desc_len_from_data(na, nb, len(edges), len(ka), len(kb), edges, b)
        print("biparite: {}".format(dl))
    else:
    	is_bipartite = False
    	dl = get_desc_len_from_data_uni(n, len(edges), B, edges, b)
    	print("non-bipartite: {}".format(dl))
    dl_B_pair += [(dl, B, len(ka), len(kb), is_bipartite)]
print(sorted(dl_B_pair, key=lambda x: x[0]))
print("ratio that successfully finds bipartite structure: {}".format(counts/100.))
