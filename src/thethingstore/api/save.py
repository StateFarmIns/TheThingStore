"""Implement saving routines.

This allows for modestly intelligent representation of multiple
types of data.
"""

import os
import pickle  # nosec
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from thethingstore.api._fs import get_fs
from pyarrow.fs import FileSystem
from typing import Any, Optional

####################################################################
#                      Conditional Imports                         #
# ---------------------------------------------------------------- #
# These are here to provide utility *if needed*.                   #
####################################################################

try:
    # flake doesn't really like these conditional imports
    # and that's okay
    from sklearn.base import BaseEstimator  # noqa: F401
    from sklearn.neighbors import BallTree  # noqa: F401
except ImportError:
    # Do nothing.
    _ = False

try:
    import joblib  # noqa: F401
except ImportError:
    # Do nothing.
    _ = False

atomic_types = (str, int, float)


def _is_atomic(x: Any) -> bool:
    return isinstance(x, atomic_types)


def _is_pd_series(x: Any) -> bool:
    try:
        import pandas as pd
    except BaseException:
        return False
    return isinstance(x, pd.Series)


def _is_pd_frame(x: Any) -> bool:
    try:
        import pandas as pd
    except BaseException:
        return False
    return isinstance(x, pd.DataFrame)


def _is_np_array(x: Any) -> bool:
    try:
        import numpy as np
    except ImportError:
        return False
    return isinstance(x, np.ndarray)


def _is_sklearn_estimator(x: Any) -> bool:
    try:
        # Check that sklearn was successfully imported
        _ = BaseEstimator
        _ = BallTree
    except NameError:
        return False
    return isinstance(x, (BaseEstimator, BallTree))


def _is_torch(x: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(x, torch.Tensor)


def _table_from_series(x: Any) -> pa.Table:
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError
    # Writing them in as columns makes the index names
    #   available as schema.
    # This method strips the index.
    return pa.Table.from_pandas(pd.DataFrame({x.name: x}))


def _np_save(x: Any, flpath: str, fs: FileSystem) -> None:
    try:
        import numpy as np
    except ImportError:
        raise RuntimeError
    with fs.open_output_stream(flpath) as f:
        np.save(f, x)


def _joblib_save(x: Any, flpath: str, fs: FileSystem) -> None:
    try:
        import joblib  # noqa: F811
    except ImportError:
        raise RuntimeError
    with fs.open_output_stream(flpath) as f:
        joblib.dump(x, f)


def _torch_save(x: Any, flpath: str, filesystem: FileSystem) -> None:
    try:
        import ibis
    except ImportError:
        raise RuntimeError
    pq.write_table(
        ibis.memtable(x).to_pyarrow(),
        flpath,
        filesystem=filesystem,
    )


def save(  # noqa: C901
    thing: Any, location: str, filesystem: Optional[FileSystem] = None
) -> None:
    """This implements a central wrapper for saving things.

    This saves your thing (hopefully intelligently.)

    If you have a data structure which can be mapped into a parquet
    document effectively, this will do that. If it cannot map that
    effectively then it will save it via pickle. Lists and mapping
    will be breadth-first expanded if they are encountered and contain
    more than 'atomic' types.

    This saves individual things, a sequence of things, or
    a mapping of things.

    To load those things back into RAM you should be calling materialize
    (`/src/thethingstore/api/load.py`).

    To see a good number of examples, what they look like represented
    on disk, and what it looks like when they're returned please view
    the [tests](tests/test_the-thing-store/test_item_save_and_load.py)

    This does not respect pandas indices. If those are important to you
    then you need to account for them explicitly, or file an issue.

    Parameters
    ----------
    thing: Any
        The thing you want to save.
    location: str
        The folder you wish it placed in.
    filesystem: Optional[FileSystem] = None
        If you want to provide a filesystem...
    """
    if filesystem is None:
        location, filesystem = get_fs(location)  # type: ignore
    if not isinstance(location, str):
        raise NotImplementedError("Save currently only saves singular paths.")
    prefix, item = os.path.split(location)
    file_info = filesystem.get_file_info(location)
    if file_info.type.name == "Directory":
        raise RuntimeError(f"Cannot save on top of an existing directory ({location}).")
    if _is_atomic(thing):
        pq.write_table(
            pa.Table.from_pylist([{"item": thing}]),
            os.path.join(prefix, f"ts-atomic-{item}.parquet"),
            filesystem=filesystem,
        )
    elif _is_pd_series(thing):
        pq.write_table(
            _table_from_series(thing),
            os.path.join(prefix, f"ts-series-{item}.parquet"),
            filesystem=filesystem,
        )
    elif _is_pd_frame(thing):
        pq.write_table(
            pa.Table.from_pandas(thing),
            os.path.join(prefix, f"ts-dataset-{item}.parquet"),
            filesystem=filesystem,
        )
    elif isinstance(thing, ds.Dataset):
        ds.write_dataset(
            thing, base_dir=location, format="parquet", filesystem=filesystem
        )
    elif isinstance(thing, pa.Array):
        pq.write_table(
            pa.Table.from_arrays([thing], names=["data"]),
            os.path.join(prefix, f"ts-pa-{item}.parquet"),
            filesystem=filesystem,
        )
    elif isinstance(thing, pa.Table):
        pq.write_table(
            thing,
            os.path.join(prefix, f"ts-dataset-{item}.parquet"),
            filesystem=filesystem,
        )
    elif isinstance(thing, dict):
        # If all the things in here are ints, floats, strings, e.g.
        if all(isinstance(_, atomic_types) for _ in thing.values()):
            # Writing them in as columns makes the index names
            #   available as schema.
            pq.write_table(
                pa.Table.from_pylist([{str(k): v for k, v in thing.items()}]),
                os.path.join(prefix, f"ts-dict-{item}.parquet"),
                filesystem=filesystem,
            )
        else:
            # Create a folder for the dictionary and try again
            tmppath = os.path.join(prefix, f"ts-dict-{item}")
            os.makedirs(tmppath, exist_ok=True)
            for k, v in thing.items():
                save(v, os.path.join(tmppath, str(k)), filesystem=filesystem)
    elif isinstance(thing, list):
        if all(isinstance(_, atomic_types) for _ in thing):
            pq.write_table(
                pa.Table.from_pylist([{str(i): v for i, v in enumerate(thing)}]),
                os.path.join(prefix, f"ts-list-{item}.parquet"),
                filesystem=filesystem,
            )
        else:
            # Create a folder for the list and try again
            tmppath = os.path.join(prefix, f"ts-list-{item}")
            os.makedirs(tmppath, exist_ok=True)
            for i, v in enumerate(thing):
                save(v, os.path.join(tmppath, f"{i}"), filesystem=filesystem)
    elif _is_np_array(thing):
        _np_save(thing, os.path.join(prefix, f"ts-numpy-{item}.npy"), filesystem)
    elif _is_sklearn_estimator(thing):
        _joblib_save(
            thing, os.path.join(prefix, f"ts-skmodel-{item}.joblib"), filesystem
        )
    elif _is_torch(thing):
        _torch_save(thing, os.path.join(prefix, f"ts-torch-{item}.parquet"), filesystem)
    else:  # If all else fails.
        with filesystem.open_output_stream(
            os.path.join(prefix, f"ts-thing-{item}.pickle")
        ) as f:
            pickle.dump(thing, f)
