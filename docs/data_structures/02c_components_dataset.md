# Component Expectations

The ThingStore manages a set of Things, each of which has component level expectations.

## Dataset

A Thing can have a tabular Dataset component which is both *identifiable* and *retrievable*.

A dataset is a collection of records; a dataset can be a small collection (like my bank records for a month) or a large collection (like a particular business problem.)

* Identity: An identity is small and cheap to hand around, like a filepath.
* Representation: A representation is full and explicit, such as a DataFrame.

### Dataset Identification

A dataset can be identified in many ways, though some of the most convenient ways of identifying data include:
* A string which is representative of a file to be loaded. This is an atomic value.
* A complex data structure which is representative of labeled partitions of data.

Here are some concrete examples which you could use in Python. Note that all these examples use parquet. This is intentional. Please reach out to data stewards for more information.

**Atomic examples**

```python
parquet_filepath_1 = 'myfolder/myfile.parquet'
parquet_filepath_2 = 's3://my/most/favoritest/prefix/whateverfolderfullofparquet'
fileid = 'fileid://whateverstuff'
```

**Complex examples**

```python
parquet_filepaths ={'2028 data': 'myfolder/data1.parquet', '2029 data': 'myfolder/data2.parquet'}
fileids ={'2028 data': 'fileid://whateverstuff#1', '2029 data': 'fileid://whateverstuff#2'}
```

Both of these examples identify datasets, though the complex examples identify *partitioned* datasets

### Dataset Representation

Datasets should be representable, meaning that data can be retrieved and brought into RAM (either whole or piece-meal.)

Data can be delayed or actualized, meaning that you can use a Dataset Identifier to point out data which may be loaded, or you can pull the data into scope.

Datasets are simply retrievable.

**Atomic examples**

```python
import pandas as pd

df_1 = pd.read_parquet(parquet_filepath_1)
df_2 = pd.read_parquet(parquet_filepath_2)

# Assume you have a ThingStore spun up and instantiated

df_fileid = my_thing_store.load(fileid)
```

**Complex examples**

```python
parquet_filepaths ={'2028 data': 'myfolder/data1.parquet', '2029 data': 'myfolder/data2.parquet'}
fileids ={'2028 data': 'fileid://whateverstuff#1', '2029 data': 'fileid://whateverstuff#2'}

import pandas as pd
import pyarrow.dataset as ds

df_all = pd.concat(
    [
        pd.read_parquet(v).assign(PARTITION=k)
        for k, v in parquet_filepaths.items()
    ]
)

# Assume you have a ThingStore spun up and instantiated
# This doesn't set a partition field in this manner,
#   but still vertically concatenates.

ds_all = ds.dataset(
    [
        my_thing_store.get_dataset(v)
        for k, v in fileids
    ]
).to_pandas()
```