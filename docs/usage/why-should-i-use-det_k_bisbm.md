# Why should I use det_k_bisbm?

`det_k_bisbm` is a replacement for [Graph-tool](https://graph-tool.skewed.de/)'s [`minimize_blockmodel_dl`](https://graph-tool.skewed.de/static/doc/inference.html#graph_tool.inference.minimize.minimize_blockmodel_dl) function with the following advantages:

* control directly the numbers of communities to infer for a bipartite network. There are 2 numbers that we can specify, one for each node type.

* conclude a different partition with a smaller description length (and a higher AMI on tested synthetic dataset).

