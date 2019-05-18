## Explore the consistency of results

As we cannot reach the globally optimal partition, we may want to check the consistency of results from run to run.

For example, we might want to calculate the description length of the data at a single point, `(ka, kb)`, 
repeatedly, without running through the whole heuristic. One can use,

```python
oks.compute_and_update(ka, kb, recompute=True)
```

We then check an internal variable, which faithfully stores the description length (or entropy) that we have computed,

```python
oks.bookkeeping_dl[(ka, kb)]
```

We have a bookkeeping of other useful observables, too. They are `bookkeeping_dl`, `bookkeeping_e_rs`, and `trace_mb`.
When we run this repeatedly, we are accessing the precision of the inference engine. 
Note that unless the underlying graph is super structured (e.g., bipartite cliques),
the resulting description lengths vary.
