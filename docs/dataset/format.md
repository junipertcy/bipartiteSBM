# Format
This program accepts input data as a text file of graph adjacencies, say `graph.edgelist`, which contains one edge per line. Each line follows an out-neighbor adjacency list format; that is, a 2-tuple of node indexes of the form,
```ini
<node_source_id_1> <out_neighbor_id_1>
<node_source_id_1> <out_neighbor_id_2>
...
<node_source_id_1> <out_neighbor_id_<outdegree>>
...
```
The `get_edgelist` function can resolve text files with custom delimiters.