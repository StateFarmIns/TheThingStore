# Component Expectations

The ThingStore manages a set of Things, each of which has component level expectations.

## Parameters

A Thing can have Parameters.

Parameters, in plain English, can be thought of as either:
* These are the levers that can be changed with this Thing (function which can be called), or
* These are the levers that were used with this Thing (parameters of function that **was called**).

Parameters for any given process take the form of a key-value mapping of elements; those elements must be atomic but may be nested.

If a set of parameters, or a single value in a set of parameters, are represented as a fileid the parameters component for the referred Thing are assumed to have a compatible structure. The parameters for that fileid are used as-is.

This allows for potentially deeply nested hierarchical sets of workflow parameters.

Appropriately designed **atomic workflows do not use nested parameters**. Workflows which reuse separate components highly probably do use nested parameters, as each workflow represents a point of potential parameter update.

Parameters are used to modify the behavior when calling a function. There is an assumption that modifying a parameter changes the output of the function, implying that everything relying on the output will *also* change.

Here are some concrete examples for a [decision tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) which you could use in Python.

```python
standard_business_appropriate_dtree_parameters = {
    'criterion': 'gini,
    'max_depth': 2,
}

param_fileid = my_thing_store.log(
    parameters=standard_business_appropriate_dtree_parameters
)

my_thing_store.get_parameters(param_fileid) == standard_business_appropriate_dtree_parameters

new_params = {
    'model': 'specific_d_tree_implementation',
    'model_params': param_fileid
}

new_params_fileid = my_thing_store.log(
    parameters=new_params
)

param_fileid_ref = my_thing_store.get_parameters(
    new_params_fileid
)['model_params']

param_fileid_ref == param_fileid
```
