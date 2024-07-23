```python
pip install 'ibis-framework[duckdb]' -q
```

    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.



```python
pip install -e .[dev] -q
```

    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.



```python
pip install torch -q
```

    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.



```python
from thethingstore import FileSystemThingStore
from pyarrow.fs import LocalFileSystem
import tempfile
import torch
import numpy as np
import pyarrow as pa

# Create a torch tensor
import torch
from numpy.random import default_rng

with tempfile.TemporaryDirectory() as t:
    data_layer = FileSystemThingStore(
        managed_location=t,
        metadata_filesystem=LocalFileSystem()
    )

    rng = default_rng()
    dims = [(200, 17), (10, 1), (1,1000)]
    for dim in dims:
        tt = torch.tensor(rng.random(dim))
        fl = data_layer.log(embedding=tt)
        import ibis
        # embed = ibis.memtable(
        embed = data_layer.get_embedding(fl, output_format="table")
        # ).to_torch()
        assert np.allclose(tt, embed)
```

    No file @/tmp/tmpim0hwbh0/metadata.parquet
    Default file with schema (FILE_ID: string
    FILE_VERSION: int64) created @/tmp/tmpim0hwbh0/metadata.parquet
    No file @/tmp/tmpim0hwbh0/metadata-lock.parquet
    Default file with schema (USER: string) created @/tmp/tmpim0hwbh0/metadata-lock.parquet

