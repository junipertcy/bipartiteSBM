Why should I use bipartiteSBM?
==============================

The :mod:`biSBM` is a replacement for `Graph-tool <https://graph-tool.skewed.de/>`_'s
`minimize_blockmodel_dl <https://graph-tool.skewed.de/static/doc/inference.html#graph_tool.inference.minimize.minimize_blockmodel_dl>`_ function,
specifically tailored for bipartite networks, with the following advantages:

* control directly the numbers of communities to infer for a bipartite network. There are 2 numbers that we can specify; i.e., :math:`K_a` and :math:`K_b`, one for each node type.

* conclude a different partition with a smaller description length (and a higher AMI on tested synthetic dataset).

And, similar to `minimize_blockmodel_dl <https://graph-tool.skewed.de/static/doc/inference.html#graph_tool.inference.minimize.minimize_blockmodel_dl>`_
and `minimize_nested_blockmodel_dl <https://graph-tool.skewed.de/static/doc/inference.html#graph_tool.inference.minimize.minimize_nested_blockmodel_dl>`_,
it shares with many of their good properties:

* converge to consistent partitions.

* estimate the SBM parameters parsimoniously (without over-fitting or under-fitting).

However, there are also some disadvantages of this program:

* It's slower than `minimize_blockmodel_dl <https://graph-tool.skewed.de/static/doc/inference.html#graph_tool.inference.minimize.minimize_blockmodel_dl>`_.

* It is not guaranteed to find the globally optimal partition.
