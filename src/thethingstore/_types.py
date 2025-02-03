"""Contains common types and type utilities.

This uses conditional imports to build sets of implemented types.

This sets out types for the different metadata components, defines
some typing for the different kinds of datasets used in the TS, and
exposes a few utilities for dataset and component tying representation.

Dataset Typing
~~~~~~~~~~~~~~
* InMemoryDatasets: This is a mapping from the string name of the
    dataset representation (i.e. 'PyArrowTable') to the class used
    in typing (i.e. `pyarrow.Table`) for *those representations
    which consume potentially large amounts of ram* and represent
    in-memory data.
* OnDiskDatasets: This is a mapping from the string name of the
    dataset representation (i.e. 'ParquetDocument') to the class used
    in typing (i.e. `str`) for those representations which are
    'ticket' (or memory efficient) representations of remote data.

Typing Utilities
~~~~~~~~~~~~~~~~
* get_info: Call this on a dict / list / set of strings
    (heirarchical or no) to get an identically shaped by-name
    representation of the data.
* _unique: A hidden method, intended to be used with the output of
    get_info, which represents the **unique types** in the output.
    If one potentially deeply nested mapping has both parquet and
    shape held within this will have two values.
"""

import datetime
import logging
import os

from thethingstore.api import error as tse
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    TypedDict,
    Type,
    Union,
)
from pyarrow import Table
from pyarrow.dataset import Dataset as paDataset
from pandas import DataFrame as pdDataFrame

logger = logging.getLogger(__name__)

####################################################################
#             Atomic Typing and Conditional Imports                #
# ---------------------------------------------------------------- #
# The TS is intended to be installed and used in a variety of env  #
#   with a variety of tools which it supports. Type checking all   #
#   these potential bits is challenging and so you see these       #
#   conditional imports used. They are reused throughout the code  #
#   base.                                                          #
####################################################################

try:
    from geopandas import GeoDataFrame
except ImportError:
    GeoDataFrame = str
    # Do nothing. This is only for the testing framework.
    _ = False


#######################
# Metadata Components #
#######################
# TODO: Make Param / metadata more arbitrary.
FileId = str
Address = str
Atomic = Union[str, int, float, datetime.datetime, FileId, Type[None]]
Complex = Union[Iterable[Atomic], Mapping[str, Atomic]]
Dataset = Union[Atomic, Complex, paDataset, Table, pdDataFrame, GeoDataFrame]
Metadata = Atomic  # type: ignore
Parameter = Union[Atomic, Complex]
Metric = Atomic  # type: ignore


class Thing(TypedDict, total=False):
    """Represent a Thing."""

    dataset: Optional[Dataset]  # type: ignore
    metadata: Optional[Mapping[str, Metadata]]  # type: ignore
    parameters: Optional[Mapping[str, Parameter]]  # type: ignore
    embedding: Optional[Dataset]  # type: ignore
    metrics: Optional[Mapping[str, Metric]]  # type: ignore
    artifacts: Optional[Mapping[str, Any]]
    function: Optional[Callable]


###############
# Data Layers #
###############
# This is a thing that implements the ThingStore API
DataLayer = Any

###########################
# Dataset Representations #
###########################
# TODO: Add PySpark Dynamic DataFrame Here
InMemoryDatasets = {
    "PyArrowTable": Table,
    "PandasDataFrame": pdDataFrame,
    "GeoDataFrame": GeoDataFrame,
}
OnDiskDatasets = {
    "ParquetDocument": str,
    "ShapeDocument": str,
    "PickleDocument": str,
    "PyArrowDataset": str,
    "FileID": str,
    "JSONDocument": str,
}


####################################################################
#                          Typing Utilities                        #
# ---------------------------------------------------------------- #
# These utilities defined below are intended to be reused through  #
#   the code base.                                                 #
####################################################################
# This is a 'string, or a list of string, or a dict of str'
_info_out_types = Union[str, List["_info_out_types"], Dict[str, "_info_out_types"]]

_info_header = "[Data Type Error]:"


def get_info(obj: Any) -> _info_out_types:
    """Return type of thing.

    This can be called on filepaths, or on objects.
    This can determine whether you are using a filepath, a PyArrow
    table, a PyArrow Dataset, a File ID, or a few different things.

    This can be called on singletons, or it can be called on deeply
    nested container representations of singletons.

    Parameters
    ----------
    obj: Any
        This is a thing. What is it? Who knows!?

    Returns
    -------
    type_of_thing: _info_out_types
        The type of the thing.

    Examples
    --------
    This is how to use this on parquet files.
    >>> get_info('silly_thing.parquet')
    'ParquetDocument'
    >>> get_info(['silly_thing1.parquet', 'silly_thing2.parquet'])
    ['ParquetDocument', 'ParquetDocument']
    >>> get_info(['s3://bucket/prefix/silly_thing1.parquet', 'silly_thing2.parquet'])
    ['ParquetDocument', 'ParquetDocument']

    While this is what it looks like for datasets.
    >>> import pyarrow.dataset as ds
    >>> _dataset = ds.dataset('tests/test_data/sample.parquet')
    >>> get_info(_dataset)
    'PyArrowDataset'
    >>> get_info([_dataset, _dataset])
    ['PyArrowDataset', 'PyArrowDataset']

    And shapes too!
    >>> get_info('sillything.shp')
    'ShapeDocument'
    """
    # Container types
    if isinstance(obj, dict):
        return {k: get_info(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [get_info(v) for v in obj]
    elif isinstance(obj, paDataset):
        return "PyArrowDataset"
    elif isinstance(obj, Table):
        return "PyArrowTable"
    elif isinstance(obj, pdDataFrame):
        return "PandasDataFrame"
    elif isinstance(obj, str):
        return _get_str_info(obj)
    else:
        raise tse.ThingStoreTypeError(type(obj))


def _get_str_info(  # noqa: C901 - This is as complex as it needs to be.
    obj: str,
) -> _info_out_types:
    """Return 'type' of string."""
    _obj_lower = obj.lower()
    if _obj_lower.startswith("fileid://"):
        return "FileID"
    elif _obj_lower.endswith(".parquet"):
        return "ParquetDocument"
    elif _obj_lower.endswith(".shp"):
        return "ShapeDocument"
    elif _obj_lower.endswith(".shape"):
        return "ShapeDocument"
    elif _obj_lower.endswith(".gdb"):
        return "ShapeDocument"
    elif _obj_lower.endswith(".pickle"):
        return "PickleDocument"
    elif _obj_lower.endswith(".pkl"):
        return "PickleDocument"
    elif _obj_lower.endswith(".json"):
        return "JSONDocument"
    else:
        if os.path.exists(obj) and os.path.isdir(obj):  # This is a local dir.
            return [_get_str_info(os.path.join(obj, _)) for _ in os.listdir(obj)]
        elif not len(os.path.splitext(obj)[1]):  # No file extension
            # It's *maybe*? a file id.:
            err_msg = _info_header + f"Dataset Type Not Identifiable - {obj}"
            logger.warning(err_msg + "Assumed FileID")
            return "FileID"
        else:
            raise tse.ThingStoreLoadingError(
                "".join(
                    [
                        _info_header,
                        f"Cannot understand dataset with suffix {os.path.splitext(obj)[1]}",
                    ]
                )
            )


def _unique(all_info: _info_out_types) -> set:
    """Build sets of types.

    This is intended to be called on the OUTPUT of get_info and is
    not a public method.
    """
    if isinstance(all_info, str):  # Only one string
        # It is a set by itself.
        return {all_info}
    elif isinstance(all_info, list):
        # Recursively unroll lists
        # Note the *tuple unpacking into union.
        return set.union(*[_unique(_) for _ in all_info])
    elif isinstance(all_info, dict):
        # Recursively unroll values of dicts.
        # Note the *tuple unpacking into union.
        return set.union(*[_unique(_) for _ in all_info.values()])
    else:
        raise NotImplementedError("get_info output **only**.")
