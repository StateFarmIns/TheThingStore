# Thing Store Components and Primary Data Structures

## Thing Components

All Things have metadata; they also have additional [*components*](src\thethingstore\thing_components.py).

* Dataset
* Metrics
* Parameters
* Metadata
* Artifacts
* Functions
* Embedding

## Thing Store

A Thing Store is another **[formal data structure](src\thethingstore\thing_store_base.py)** which potentially contains one or more Things, potentially dependent on one another.

## Thing Node / Graph

Any particular Thing can *always* be viewed as a [Node](src\thethingstore\thing_node.py) in a larger [Graph](src\thethingstore\thing_graph.py).

## Metadata Dataset

The default schema for any Thing Store node includes elements built solely for the Thing Store (primarily process tracking components) and customer desired elements. All Things are recorded in the Metadata Dataset with (at least) the default metadata and any specific metadata logged with individual Things.

## FileID

A FileID is a unique identifier used to tag everything that's logged in the Thing store. It may be relative (on my local filesystem), or absolute (this particular S3 bucket / prefix.)

Is this Thing in a location where people are going to be able to use it? If it's on your machine, probably not. If it's on a remote machine, maybe?

Having the capability to say 'this is how I store the data' in an efficient way makes it easier to save and load efficient and reusable representations for data in automation and for reuse.

Check out the documentation and examples in the [source](src\thethingstore\file_id.py) (not very complex) for a good technical understanding of what the fileid file schema is.
