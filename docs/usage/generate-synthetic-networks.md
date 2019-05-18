# Generate synthetic networks

It is necessary to benchmark our algorithm via synthetic networks, where we have perfect confidence 
on the model parameters. Here we can control the nodes' block membership `b`, the edge count matrix `e_rs`, 
and the degree distribution `d`. Once we specified these 3 parameters, we call [Graph-tool](
https://graph-tool.skewed.de/)'s [`generate_sbm`](
https://graph-tool.skewed.de/static/doc/generation.html#graph_tool.generation.generate_sbm) function to
generate a graph-tool graph instance, and then use that instance as an input for our `OptimalKs` class.

## Block membership

## Edge count matrix
### Easy cases
The easy cases are like planted partition models, but in the bipartite form, where `Ka = Kb`.


### Hard cases
The hard cases are planted networks where `Ka != Kb`, but the edge count matrix is still
designed in a way that allow the control of network structural strength.


### Even harder cases
A even harder case is simply a random draw out of the ensemble of edge count matrices where we 
only fix `Ka` and `Kb`. 

## Degree distribution
