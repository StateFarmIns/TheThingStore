# TODO

You've heard of an [object store](https://cloud.google.com/learn/what-is-object-storage), you've heard of a [feature store](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/), you might have even heard of an [embedding / vector store](https://www.embedding.store/), but what's next?

What about anything?

## The 'Thing' Store

Look, you've got something that you think is important and you might want to reuse it.

Maybe you want to store that in s3. Maybe you want to store it in Athena. Maybe you want to store that on your local computer.

Does it matter?

You should be able to just say 'give me my thing' and get it back, regardless of what it looks like.

How do you do that?

You say, that for any *Thing*, it might have a few important flavors of information:

* Metadata: You might have some labels that you think are appropriate. 'Funny=True', or cat color is green, or whatever. This is a mapping of keys to values.
* Parameters: This thing will use these to run, or *did* use these to run.
* Metrics: This thing measured these metrics.
* Dataset: This thing is associated with a view of a tabular dataset.
* Artifacts: I need these files, too.
* Embedding: This thing is represented with this embedded representation. (TODO)
* Function: This thing is represented with some functionality. (TODO)

## Development Plan

1. Get devcontainer up and running.
2. Get pipeline up and running.
3. Produce documentation.
4. Split branches for each implementation.
5. Complete CRUD interface for each implementation
6. Deploy interface for Data Management
7. Retool the interface to serve via pyarrow streaming parquet lake, regardless.
8. Fully bake function and embedding components
9. Alternative backend implementations and improvements