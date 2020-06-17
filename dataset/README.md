## Empirical datasets

All empirical datasets (except for the IMDb one) can be downloaded from the [Colorado Index of Complex Networks (ICON)](https://icon.colorado.edu/).
The processed `edgelist` *.txt files can be downloaded from [this link](https://bag.netscied.tw/s/Mc8JHCaTd3nWa36). 

The filename follows this convention:
  `{title}-{entity_name_for_type_a}_{n_a}-{entity_name_for_type_b}_{n_b}.edgelist`
  
With `n_a` and `n_b` being the number of nodes in type-_a_ and type-_b_, respectively.

The node indexes follow as:

| type-_a_             | type-_b_                             | 
| -------------------- |:------------------------------------:| 
| 0, 1, ..., (n_a - 1) | n_a, (n_a + 1), ..., (n_a + n_b - 1) | 
