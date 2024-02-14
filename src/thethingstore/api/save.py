"""Implement saving routines.

This allows for modestly intelligent representation of multiple
types of data.
"""

import numpy as np
import os
import pandas as pd
import pickle  # nosec - This requires explicit loading.
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
    from sklearn.base import BaseEstimator
    from sklearn.neighbors import BallTree
except ImportError:
    # Do nothing.
    _ = False

try:
    import joblib
except ImportError:
    # Do nothing.
    _ = False

atomic_types = (str, int, float)


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
    if isinstance(thing, (float, int, str)):
        pq.write_table(
            pa.Table.from_pylist([{"item": thing}]),
            os.path.join(prefix, f"ts-atomic-{item}.parquet"),
            filesystem=filesystem,
        )
    elif isinstance(thing, pd.Series):
        # Writing them in as columns makes the index names
        #   available as schema.
        # This method strips the index.
        pq.write_table(
            pa.Table.from_pandas(pd.DataFrame({thing.name: thing})),
            os.path.join(prefix, f"ts-series-{item}.parquet"),
            filesystem=filesystem,
        )
    elif isinstance(thing, pd.DataFrame):
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
    elif isinstance(thing, np.ndarray):
        with filesystem.open_output_stream(
            os.path.join(prefix, f"ts-numpy-{item}.npy")
        ) as f:
            np.save(f, thing)
    elif isinstance(thing, (BaseEstimator, BallTree)):
        with filesystem.open_output_stream(
            os.path.join(prefix, f"ts-skmodel-{item}.joblib")
        ) as f:
            joblib.dump(thing, f)
    else:  # If all else fails.
        with filesystem.open_output_stream(
            os.path.join(prefix, f"ts-thing-{item}.pickle")
        ) as f:
            pickle.dump(thing, f)
