# Functionality in the ThingStore

If you're an engineer, scientist, or analyst, you're probably familiar with Python.

You probably use Python, a lot.

This is an example of how you can store and reuse Python functionality within the Thing Store.

Note that the Thing Store **does not take care of the environmental requirements**.

You *can*, though, attach a `requirements.txt` in the artifacts of your things and install that as you desire prior to calling the function.

**This** specific example uses *this specific function*:


```python
!cat test_function.py
```

    """Example Testing Function.
    
    This has every component.
    """
    import pandas as pd
    from thethingstore import ThingStore as DataLayer
    from thethingstore.types import FileId, Thing
    
    workflow_metadata = {'example': 'label', 'FILE_ID': 'EXTREMELY_SPECIFIC'}
    workflow_artifacts = {'inmemory': 'stuff'}
    # workflow_embedding = [1., 2., 3.]  # Not implemented just yet.
    # workflow_embedding = [[1., 2., 3.], [4., 5., 6.]]  # Not implemented just yet.
    workflow_dataset = pd.DataFrame({'stupid': [1., 2., 3.]})
    workflow_metrics = {'metric': 'SPECIFIC_MEASUREMENT_FUNCTION'}
    
    
    def workflow(number_fileid: FileId = 'asd') -> Thing:
        return {
            'metadata': {'mew': 'labels'}
        }

The contents of the function are meaningless, but they allow you to see how a functional Thing is built.

* Most Components: If there are variables such as `workflow_{component}` excluding function and parameters they are scraped from the code, itself, when the function is published.
* Parameters: The parameters are built from the function signature.
* Function: The **workflow** function (naming requirement) identifies the function to be imported and returned.
* Return: The workflow returns a **Thing**. This isn't a requirement, but is a strong suggestion.

## Saving that function to the ThingStore

Here is a quick example of how to save the function to the ThingStore. Note this loading function is friendly enough that it can help you understand when a requirement isn't met.


```python
from pyarrow.fs import LocalFileSystem
from thethingstore.thing_store_log import log_function
from thethingstore import FileSystemThingStore

temp_layer = FileSystemThingStore(
    metadata_filesystem=LocalFileSystem(),
    managed_location='managed_files'
)

log_function(
    thing_store=temp_layer,
    python_file_path='test_function.py',
    dry_fire=True
)

log_function(
    thing_store=temp_layer,
    python_file_path='test_function.py',
    dry_fire=False
)
```

    No file @/tmp/tmp2m2npn_5/thingstore/metadata.parquet
    Default file with schema (FILE_ID: string
    FILE_VERSION: int64) created @/tmp/tmp2m2npn_5/thingstore/metadata.parquet
    No file @/tmp/tmp2m2npn_5/thingstore/metadata-lock.parquet
    Default file with schema (USER: string) created @/tmp/tmp2m2npn_5/thingstore/metadata-lock.parquet
    No file @/tmp/tmpgb_anc7m/thingstore/metadata.parquet
    Default file with schema (FILE_ID: string
    FILE_VERSION: int64) created @/tmp/tmpgb_anc7m/thingstore/metadata.parquet
    No file @/tmp/tmpgb_anc7m/thingstore/metadata-lock.parquet
    Default file with schema (USER: string) created @/tmp/tmpgb_anc7m/thingstore/metadata-lock.parquet
    FILE_VERSION ALREADY EXISTS: AMENDED TO 4





    'EXTREMELY_SPECIFIC'



## Calling that function from the ThingStore

Here is a quick example of how to pull down the function from the ThingStore and run it.


```python
f = temp_layer.get_function('EXTREMELY_SPECIFIC')
```


```python
f()
```




    {'metadata': {'mew': 'labels'}}


