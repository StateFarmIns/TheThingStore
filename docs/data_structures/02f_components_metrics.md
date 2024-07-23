# Component Expectations

The ThingStore manages a set of Things, each of which has component level expectations.

## Metrics

A Thing can have *Metrics*.

Metrics in general can be considered 'how *good* is this Thing' and could be measuring (among many others):
* Accuracy (of a classifier)
* Number of bytes (required to load a dataset)
* Processing time (required to 'do the Thing')

Metrics for any given Thing take the form of a key-value mapping of atomic elements.

If a single metric is represented as a fileid that metric is assumed to have a compatible *function* which is intended to be automatically run to measure the output of this fileid and can be loaded from the data layer.

If a single metric is represented as a float or integer value that metric is assumed to be direct measurement outcome.

Here are some concrete examples which you could use in Python.

```python
metrics = {
    'silly': 1,
    'example': 2.3,
    'impurity': 0.0001
}
metric_fileid = my_thing_store.log(metrics=metrics)
```

## Metrics and Functional Things

Functions, specifically, don't typically measure the source code (though it's a good idea) and so if you have a Functional Thing and it has a Metric, that Metric is *assumed to be* a function that may be called and executed on the outcome of the Functional Thing.

This will be formally codified with `.run` when it is developed.
