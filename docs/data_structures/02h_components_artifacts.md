# Component Expectations

The ThingStore manages a set of Things, each of which has component level expectations.

## Artifacts

A Thing can have Artifacts.

What is an artifact? Everything that doesn't fit neatly somewhere else.

Ball trees, html reports, logs, provisioning artifacts, etc. These are all valid files for retention that could be attached to inform processes.

Artifacts may be identified as a string; that string is assumed to either be an accessible folderpath or a file identifer which contains a representative set of artifacts to reuse.

Here is a concrete example you could use in Python.


```python
>>> import os
>>> import tempfile
>>> from thethingstore import FileSystemThingStore
>>> from thethingstore.thing_store_pa_fs import pyarrow_tree
>>> from pprint import pprint
>>> from pyarrow.fs import LocalFileSystem

>>> with tempfile.TemporaryDirectory() as t:    
...     my_stupid_text = '''I have words.
...     So many words.
...     Too many words...
...     '''
...     my_thing_store = FileSystemThingStore(
...         managed_location=os.path.join(t, 'data_layer'),
...         metadata_filesystem=LocalFileSystem()
...     )
...         
...     with open(os.path.join(t, 'file'), 'w') as f:
...         f.write(my_stupid_text)
... 
...     artifact_fileid = my_thing_store.log(
...         artifacts_folder=t
...     )
... 
...     my_thing_store.list_artifacts(artifact_fileid) == ['file']
...     
...     pprint(pyarrow_tree(path=t, filesystem=LocalFileSystem(), file_info=False))
No file @/tmp/tmpkh0z96py/data_layer/metadata.parquet
Default file with schema (FILE_ID: string
FILE_VERSION: int64) created @/tmp/tmpkh0z96py/data_layer/metadata.parquet
No file @/tmp/tmpkh0z96py/data_layer/metadata-lock.parquet
Default file with schema (USER: string) created @/tmp/tmpkh0z96py/data_layer/metadata-lock.parquet
{'data_layer': {'managed_files': {'0a93f5c177fa46aebe12d0572bca27a5': {'1': '...'}},
                'metadata-lock.parquet': 'file',
                'metadata.parquet': 'file'},
 'file': 'file'}
```

Just as a side note, you can specify artifacts instead of an artifacts_folder at log time, though that's not a documented feature. If you've read this far you deserve to know.