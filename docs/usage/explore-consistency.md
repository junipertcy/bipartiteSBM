## Explore Consistency of Results
In addition, in any case, if one wants to calculate the description length of the data at a single point, `(ka, kb)`, without running through the whole heuristic, one can use,

```python
oks.compute_and_update(ka, kb, recompute=True)
```

We then check an internal variable, which faithfully stores the description lengths that we have computed,

```python
oks.bookkeeping_dl[(ka, kb)]
```

We have a bookkeeping of other useful observables, too. They are `bookkeeping_dl`, `bookkeeping_e_rs`, and `trace_mb`.