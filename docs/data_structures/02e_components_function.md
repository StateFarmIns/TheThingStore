# Component Expectations

The ThingStore manages a set of Things, each of which has component level expectations.

## Function

A Thing can have a Function, meaning that it can be called and will produce an output.

A Functional Thing is a little bit different than a non-functional Thing; if you examine the [internal structure of a Thing](./03_graphs.md) you can see how Functional Things use their Parameters and Metrics, among other changes.

[The example within the Notebooks](../../notebooks/FunctionalThing.ipynb) is very simple to understand.

Official recommendation is 'Your function should return a Thing'.
