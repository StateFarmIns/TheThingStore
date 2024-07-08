# Simple ThingStore Application

## Turning on a ThingStore

When you turn on a Thing Store for the first time it will initialize an empty set of managed Things using a ThingStore API compliant Python interface.

Translation: If there's not a filing cabinet there for me to put stuff into, there is now.

At this point there is a metadata dataset, albeit empty, with [default schema elements](src\thethingstore\thing_store_elements.py).

## Schema Elements

The concept of a ThingStore is intended to allow you to store anything and everything in the same way.

*Everything* you store in the ThingStore will have metadata; every time you log something it will get *at least* the default schema elements. The *default* elements are 'thing store'-specific and can be very useful in many different ways.

The **non**-default elements are situationally specific and immaterial. It simply does not matter whether you use FRUIT=APPLE, FRUIT=ORANGE, COLOR=BROWN, PROCESS=TAXES, NAME=MODELING_DATASET or whatever. You simply need to be explicit in the labels which you attach to Things (ideally the labels contain useful information!)

This enables **silos** which (contrary to popular opinion) are **good** things.

Silos effectively encode expert knowledge.

Allowing people to label individual Things in individual pools with pool-specific expert knowledge (metadata) allows for semantic meaning to be explicitly encoded.

Being able to compare and contrast schema between different pools allows for semantic similarity search to be conducted and automatic labeling workflows to leverage the semantic similarity.

## Example Project In Data Pool

Within this project there is a collection of Notebooks which lay out a [lot of background](notebooks\TheLanguageOfThings_1_3.ipynb) that supports a (notional) [complex project built on top a Thing Store](notebooks\TheLanguageOfThings_1_3.ipynb).

There are examples of logging (many different styles of Things), examples of retrieving specific componets (go get me the dataset for this thing or the parameters for that thing or the metrics for those things).

There are examples of how to build and subset views of the metadata, parameters, and metrics across a large set of Things.

Stepping down into the weeds of the code that supports this Notebook, and reading the documentation for the ThingStore API (get_x, log, etc) could likely answer many how / what questions.
