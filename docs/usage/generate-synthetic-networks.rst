Generate synthetic networks
===========================

It is necessary to benchmark our algorithm via synthetic networks, where we have perfect confidence 
on the model parameters. Here we can control the nodes' block membership :math:`b`, the edge count matrix :math:`e_{rs}`,
and the degree distribution :math:`d`. Once we specified these 3 parameters,
we call `Graph-tool <https://graph-tool.skewed.de/>`_'s
`generate_sbm <https://graph-tool.skewed.de/static/doc/generation.html#graph_tool.generation.generate_sbm>`_ function to
generate a graph-tool graph instance,
and then use that instance as an input for our :class:`det_k_bisbm.optimalks.OptimalKs` class.

Block membership
----------------

Edge count matrix
-----------------

Easy cases
~~~~~~~~~~
The easy cases are like planted partition models, but in the bipartite form, where :math:`K_a = K_b`.


Hard cases
~~~~~~~~~~
The hard cases are planted networks where :math:`K_a \neq K_b`, but the edge count matrix is still
designed in a way that allow the control of network structural strength.


Even harder cases
~~~~~~~~~~~~~~~~~
A even harder case is simply a random draw out of the ensemble of edge count matrices where we 
only fix :math:`K_a` and :math:`K_b`.


Degree distribution
-------------------
