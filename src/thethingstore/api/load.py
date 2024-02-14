"""General loading utilities.

The utilities here allow for abstracting loading patterns.
"""
import collections
import logging
import numpy as np
import os
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import pathlib
import pickle  # nosec - This requires explicit loading.

from thethingstore import types as tst
from thethingstore.api import error as tsle
from thethingstore.api._fs import get_fs, ls_dir
from pyarrow.fs import S3FileSystem, FileSystem
from typing import Any, Dict, List, Iterable, Mapping, Optional, Tuple, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


try:
    import joblib
except ImportError:
    # Do nothing.
    _ = False

try:
    import geopandas as gp
except ImportError:
    # Do nothing.
    _ = False


def _get_info(  # noqa: C901
    dataset_or_filepaths: Union[tst.Dataset, List[tst.Dataset]],
) -> Tuple[Iterable[tst.Dataset], str]:
    """Coerce dataset / filepaths to list and determine type.

    This function ensures that incoming datasets are all listified
    and that the type of the items is identified. If multiple types
    are encountered this will explode.

    Parameters
    ----------
    dataset_or_filepaths: tst.Dataset
        This is a dataset, which can be one or many of many things.

    Returns
    -------
    things_to_load, dataset_type: Tuple[List[tst.Dataset], str]
        things_to_load is an iterable of the things to load.
        dataset_type can take the form 'Parquet', 'dataset',
            'table', 'pandas', 'fileid', or 'shape'.

    Examples
    --------
    This is how to use this on parquet files.
    >>> _get_info('silly_thing.parquet')
    (['silly_thing.parquet'], 'parquet')
    >>> _get_info(['silly_thing1.parquet', 'silly_thing2.parquet'])
    (['silly_thing1.parquet', 'silly_thing2.parquet'], 'parquet')

    While this is what it looks like for datasets.
    >>> import pyarrow.dataset as ds
    >>> _dataset = ds.dataset('tests/test_data/sample.parquet')
    >>> items, types = _get_info(_dataset)
    >>> len(items)
    1
    >>> types
    'dataset'
    >>> type(items[0])
    <class 'pyarrow._dataset.FileSystemDataset'>
    >>> items, types = _get_info([_dataset, _dataset])
    >>> len(items)
    2
    >>> types
    'dataset'
    >>> type(items[0])
    <class 'pyarrow._dataset.FileSystemDataset'>

    This is what it looks like for shapes.
    >>> items, types = _get_info('sillything.shp')
    >>> types
    'shape'
    >>> type(items[0])
    <class 'str'>
    """
    # TODO: Move this to string representation of class *or* something else
    #   smarter that doesn't require import of all the tools in every scope.
    # 'Duh.'
    # First, do a check to determine if this is an iterable
    #   of datasets and, if it is not, make it so.
    _atomic_type = (str, pd.DataFrame, ds.Dataset, pa.Table)
    _dataset_or_filepaths: Iterable[Any]
    if not isinstance(dataset_or_filepaths, collections.abc.Iterable):
        _dataset_or_filepaths = [dataset_or_filepaths]
    elif isinstance(dataset_or_filepaths, _atomic_type):
        _dataset_or_filepaths = [dataset_or_filepaths]
    else:
        _dataset_or_filepaths = dataset_or_filepaths
    # Now create an empty set for types.
    _types = list(set(type(_) for _ in _dataset_or_filepaths))
    # If this iterable has more than one contained type, die!
    if len(_types) > 1:
        raise tsle.ThingStoreLoadingError(
            f"""
        Loading multiple types simultaneously is not implemented!

        Types
        -----\n{_types}
        """
        )
    # Now pull out the singular type.
    _type = _types[0]
    if _type == str:
        # Dataset type can be parquet, shape, or file id
        _endswith = list({pathlib.Path(_).suffix for _ in _dataset_or_filepaths})
        if len(_endswith) > 1:
            raise tsle.ThingStoreLoadingError(
                f"""
            Loading multiple file extensions simultaneously is not implemented!

            Extensions
            ----------\n{_endswith}
            """
            )
        # If there is *no* suffix...
        if len(_endswith[0]) == 0:
            # It's *probably* a file id.
            dataset_type = "fileid"
        # Otherwise it's pretty easy to pick out.
        elif _endswith[0] == ".parquet":
            dataset_type = "parquet"
        elif _endswith[0] in (".shp", ".gdb"):
            dataset_type = "shape"
        elif _endswith[0] in (".pickle", ".pkl"):
            dataset_type = "pickle"
        else:
            raise tsle.ThingStoreLoadingError(
                f"Cannot understand dataset with suffix {_endswith[0]}"
            )
    elif _type in (ds.Dataset, ds.FileSystemDataset, ds.UnionDataset):
        dataset_type = "dataset"
    elif _type == pa.Table:
        dataset_type = "table"
    elif _type == pd.DataFrame:
        dataset_type = "pandas"
    else:
        raise tsle.ThingStoreLoadingError(f"Cannot understand dataset type {_type}")
    return _dataset_or_filepaths, dataset_type


def _s3_str_handler(
    s3path: Union[str, ds.Dataset], filesystem: Optional[S3FileSystem] = None
) -> str:
    """Turn string into S3.

    Parameters
    ----------
    s3path: Union[str, ds.Dataset]
        This might look like a filepath.
    filesystem: Optional[S3FileSystem] = None
        This could be an S3Filesystem.

    Returns
    -------
    clean_s3_path: Union[str, ds.Dataset]
        If this is a string and the filesystem is an S3FileSystem
        then the string will.

    Examples
    --------
    >>> from pyarrow.fs import S3FileSystem
    >>> _s3_str_handler('stupid/example')
    'stupid/example'
    >>> _s3_str_handler('stupid/example', filesystem='whatever')
    'stupid/example'
    >>> _s3_str_handler('stupid/example', filesystem=S3FileSystem())
    'stupid/example'
    >>> _s3_str_handler('s3://test-bucket/data/sample.parquet', filesystem=S3FileSystem())
    'test-bucket/data/sample.parquet'
    """
    # Run through a series of checks.
    if filesystem is None:
        return s3path
    if not isinstance(filesystem, S3FileSystem):
        return s3path
    if not isinstance(s3path, str):
        return s3path
    if not s3path.upper().startswith("S3://"):
        return s3path
    # At this point it's a string, we have an s3 filesystem, and the strings should look right.
    # This just makes it so we can standardize stuff!
    parsed_path = urlparse(s3path)
    return parsed_path.netloc + parsed_path.path


def _load_dataset(
    dataset_or_filepaths: Union[List[Union[str, ds.Dataset]], str, ds.Dataset],
    load_dataset_kwargs: Optional[Dict[str, Any]] = None,
) -> List[ds.Dataset]:
    """Define a dataset.

    At the end of this there is a list of datasets. Please view the
    documentation for PyArrow Dataset for the keyword arguments.

    Parameters
    ----------
    dataset_or_filepaths: List[Union[str, ds.Dataset]]
        A list of parquet or dataset files.
    load_dataset_kwargs: Optional[Dict[str, Any]] = None
        An optional dictionary of kwargs to unpack into the dataset.

    Returns
    -------
    datasets: List[ds.Dataset]
        A list of PyArrow datasets.
    """
    if not isinstance(dataset_or_filepaths, List):
        dataset_or_filepaths = [dataset_or_filepaths]
    if not all(isinstance(_, (str, ds.Dataset)) for _ in dataset_or_filepaths):
        sets = set(type(_) for _ in dataset_or_filepaths)
        raise tsle.ThingStoreLoadingError(
            f"Cannot turn item of type {sets} into dataset."
        )
    if load_dataset_kwargs is None:
        load_dataset_kwargs = {}
    # Go ahead and handle any silliness here.
    dataset_or_filepaths = [
        _s3_str_handler(_, load_dataset_kwargs.get("filesystem"))
        for _ in dataset_or_filepaths
    ]
    if isinstance(dataset_or_filepaths[0], str):
        return [ds.dataset(_, **load_dataset_kwargs) for _ in dataset_or_filepaths]
    else:
        return dataset_or_filepaths


def _load_shape(
    dataset_or_filepaths: Union[
        List[Union[str, "gp.GeoDataFrame"]], str, "gp.GeoDataFrame"
    ],
    load_pandas_kwargs: Optional[Dict[str, Any]] = None,
) -> List["gp.GeoDataFrame"]:
    """Load a shape.

    At the end of this there is a geopandas geodataframe; this uses
    GeoPandas read.

    Parameters
    ----------
    dataset_or_filepaths: List[str]
        A list of shapes.
    load_pandas_kwargs: Optional[Dict[str, Any]] = None
        An optional dictionary of kwargs to unpack into the load.

    Returns
    -------
    geodata: List[gp.GeoDataFrame]
        A list of GeoDataFrames.
    """
    try:
        import geopandas as gp
    except ImportError:
        raise Exception("Hint: pip install thethingstore[shapes] to load shapedata.")

    if not isinstance(dataset_or_filepaths, List):
        dataset_or_filepaths = [dataset_or_filepaths]
    if not all(isinstance(_, (str, gp.GeoDataFrame)) for _ in dataset_or_filepaths):
        raise tsle.ThingStoreLoadingError(
            f"Cannot turn item of type {set(type(_) for _ in dataset_or_filepaths)} into GeoPandas."
        )
    if load_pandas_kwargs is None:
        load_pandas_kwargs = {}
    if isinstance(dataset_or_filepaths[0], str):
        return [gp.read_file(_, **load_pandas_kwargs) for _ in dataset_or_filepaths]
    else:
        return dataset_or_filepaths


def _load_table(
    dataset_or_filepaths: Union[
        List[Union[ds.Dataset, pa.Table]], ds.Dataset, pa.Table
    ],
    load_table_kwargs: Optional[Dict[str, Any]] = None,
) -> List[pa.Table]:
    """Load Tables.

    At the end of this there is a List of PyArrow Tables, assumed to
    be of the same schema.

    Parameters
    ----------
    dataset_or_filepaths: List[Union[ds.Dataset, pa.Table]]
        PyArrow Datasets or tables.
    load_pandas_kwargs: Optional[Dict[str, Any]] = None
        An optional dictionary of kwargs to unpack into the load.

    Returns
    -------
    tables: List[pa.Table]
        A single schema with a list of PyArrow Tables.
    """
    if not isinstance(dataset_or_filepaths, List):
        dataset_or_filepaths = [dataset_or_filepaths]
    if not all(isinstance(_, (ds.Dataset, pa.Table)) for _ in dataset_or_filepaths):
        raise tsle.ThingStoreLoadingError(
            f"Cannot turn item of type {set(type(_) for _ in dataset_or_filepaths)} into Table."
        )
    if load_table_kwargs is None:
        load_table_kwargs = {}
    if isinstance(dataset_or_filepaths[0], ds.Dataset):
        return [_.to_table(**load_table_kwargs) for _ in dataset_or_filepaths]
    else:
        return dataset_or_filepaths


def _load_pandas(
    dataset_or_filepaths: List[Union[pa.Table, pd.DataFrame]],
    load_pandas_kwargs: Optional[Dict[str, Any]] = None,
) -> List[pd.DataFrame]:
    if not isinstance(dataset_or_filepaths, List):
        dataset_or_filepaths = [dataset_or_filepaths]
    if not all(isinstance(_, (pa.Table, pd.DataFrame)) for _ in dataset_or_filepaths):
        raise tsle.ThingStoreLoadingError(
            f"Cannot turn item of type {set(type(_) for _ in dataset_or_filepaths)} into Pandas."
        )
    if load_pandas_kwargs is None:
        load_pandas_kwargs = {}
    if isinstance(dataset_or_filepaths[0], pa.Table):
        return [_.to_pandas(**load_pandas_kwargs) for _ in dataset_or_filepaths]
    else:
        return dataset_or_filepaths


def _load_pickle(
    dataset_or_filepaths: Union[List[str], str],
    load_pickle_kwargs: Optional[Dict[str, Any]] = None,
) -> List[object]:
    trust_msg = """Pickles must be trusted.

    Pickles allow running arbitrary code. They are a potential
    vector to allow trusted agents to run malicious code. This
    function requires you to understand and explicitly acknowledge
    the risk and validate that *you know where your pickle has been
    and it can be trusted.*

    No dirty pickles are allowed.

    If you want to shut this error up please pass `trusted=True`
    in the load pickle keyword arguments.

    At the end of this there is a List of objects with no further
    assumptions.

    Parameters
    ----------
    dataset_or_filepaths: List[Union[ds.Dataset, pa.Table]]
        PyArrow Datasets or tables.
    load_pickle_kwargs: Optional[Dict[str, Any]] = None
        A required dictionary of kwargs to unpack into the load.

    Returns
    -------
    objects: List[object]
        A list of objects loaded via pickle.
    """
    if load_pickle_kwargs is None:
        _load_pickle_kwargs = {}
    else:
        _load_pickle_kwargs = load_pickle_kwargs
    if "trusted" not in _load_pickle_kwargs.keys():
        raise tsle.ThingStoreLoadingError(trust_msg)
    if not _load_pickle_kwargs.pop("trusted"):
        raise tsle.ThingStoreLoadingError(trust_msg)
    if not isinstance(dataset_or_filepaths, List):
        dataset_or_filepaths = [dataset_or_filepaths]
    if not all(isinstance(_, (str)) for _ in dataset_or_filepaths):
        raise tsle.ThingStoreLoadingError(
            f"Cannot load type {set(type(_) for _ in dataset_or_filepaths)} into pickles."
        )

    def loadit(path: str) -> Any:
        with open(path) as f:
            # This requires *explicit ackowledgement of security risk.""
            loaded_object = pickle.load(f, **_load_pickle_kwargs)  # type: ignore  # nosec
        return loaded_object

    return [loadit(_) for _ in dataset_or_filepaths]


def materialize(  # noqa: C901
    filepath: Union[str, List[str], Dict[str, Any]],
    filesystem: Optional[FileSystem] = None,
) -> Any:
    """Turn items in filesystem into an item in memory.

    These items exist in filesystem and need to be brought into memory.

    This **WILL RECURSE**.

    This requries a consistent naming convention for files.

    Parameters
    ----------
    filepath: Union[str, List[str], Dict[str, str]]
        If a string, a single file / folder.
        If a list of strings, a sequence of files / folders.
        If a dict of strings with string keys, a mapping of files / folders.

    Returns
    -------
    loaded_item: Any
        The item restored to it's former glory.
    """
    if filesystem is None:  # If no filesystem is passed we try and get one.

        filepath, filesystem = get_fs(filepath)
    if isinstance(filepath, str):  # A single thing to look at mon.
        file_info = filesystem.get_file_info(filepath)
        # Tell me what kind!
        if file_info.type.name == "NotFound":  # The file does not exist...
            raise FileNotFoundError(file_info)
        _, item = os.path.split(filepath)  # Split the file from the path.
        if file_info.type.name == "Directory":  # This is a directory!
            if item.startswith("ts-dict"):  # If it's one we manage...
                # Get all the filepaths in the directory.
                filepaths = ls_dir(filepath, filesystem=filesystem)
                # Reduce them to their names.
                filenames = [os.path.split(k)[1] for k in filepaths]
                # And strip the ts headers and filetype.
                filenames = [
                    os.path.splitext("-".join(_.split("-")[2:]))[0] for _ in filenames
                ]
                # Return that as a dictionary mapping of filename to loaded file.
                return {
                    k: materialize(v, filesystem=filesystem)
                    for k, v in zip(filenames, filepaths)
                }
            else:  # Just load all the files in a list.
                return [
                    materialize(_, filesystem=filesystem)
                    for _ in ls_dir(filepath, filesystem=filesystem)
                ]
        if not item.startswith("ts"):  # Not saved by us!
            raise Exception(
                f"""File loading error.
            This file was not saved by the Thing Store
            and cannot be understood.

            Filepath
            --------\n{filepath}
            """
            )
        if item.startswith("ts-atomic"):  # Single thing.
            return (
                ds.dataset(filepath, filesystem=filesystem)
                .to_table()
                .to_pandas()
                .iloc[0]
                .item()
            )
        elif item.startswith("ts-dict"):  # Mapping of things.
            _dataset = (
                ds.dataset(filepath, filesystem=filesystem).to_table().to_pylist()
            )
            if _dataset:
                return (
                    ds.dataset(filepath, filesystem=filesystem)
                    .to_table()
                    .to_pylist()[0]
                )
            else:
                return {}
        elif item.startswith("ts-list"):  # Sequence of things.
            return list(
                ds.dataset(filepath, filesystem=filesystem)
                .to_table()
                .to_pylist()[0]
                .values()
            )
        elif item.startswith("ts-series"):  # Sequence of things.
            return (
                ds.dataset(filepath, filesystem=filesystem)
                .to_table()
                .column(0)
                .to_pandas()
            )
        elif item.startswith("ts-numpy"):  # Sequence of things.
            with filesystem.open_input_file(filepath) as f:
                return np.load(f)
        elif item.startswith("ts-dataset"):  # Single dataset.
            return ds.dataset(filepath, filesystem=filesystem).to_table()
        elif item.startswith("ts-pa"):  # Single array.
            return (
                ds.dataset(filepath, filesystem=filesystem)
                .to_table()
                .column(0)
                .combine_chunks()
            )
        elif item.startswith("ts-skmodel"):  # A joblib model.
            with filesystem.open_input_file(filepath) as f:
                output = joblib.load(f)
            return output
        elif item.startswith("ts-thing"):  # A pickled object.
            with filesystem.open_input_file(filepath) as f:
                # TODO: Get kwargs in here?
                output = pickle.load(f)  # nosec: saved by ThingStore and trusted?
            return output
        else:  # No clue!
            raise NotImplementedError(item)
    elif isinstance(filepath, List):  # A sequence of files mon.
        if len(filepath) == 1:  # But a sequence of *one*
            return materialize(filepath[0], filesystem=filesystem)
        # Here there is a test to see if all the files follow
        #   an expected pattern.
        # This can account for ts-list and ts-dict
        if all(item.startswith("ts-dict") for item in filepath):
            output = {
                k.replace("ts-dict-", ""): materialize(k, filesystem=filesystem)
                for k in filepath
            }
        else:
            output = []
            for file in filepath:
                output.append(materialize(file, filesystem=filesystem))
            if len(output) == 1:
                output = output[0]
        return output
    elif isinstance(filepath, Dict):  # A mapping of files mon.
        # We probably don't need to replace it, but what are the
        #   odds that someone would use that string, specifically?
        # (If you're reading this... sorry!)
        output = {
            k.replace("ts-dict-", ""): materialize(v) for k, v in filepath.items()
        }
    else:
        raise NotImplementedError("ONLY STRING, LIST OF STRING, MAPPING OF STRING")


# This mapping is used to define *The Next Step* in the
#   line for each dataset. A file id, when loaded, is a path to
#   stick into read parquet.
_next_mapping = {
    "fileid": "parquet",
    "parquet": "dataset",
    "dataset": "table",
    "table": "pandas",
    "shape": "pandas",
    "pickle": "object",
}


def _map_load(
    dataset_or_filepaths: tst.Dataset,
    dataset_type: str,
    output_format: str,
    load_dataset_kwargs: Optional[Dict[str, Any]] = None,
    load_table_kwargs: Optional[Dict[str, Any]] = None,
    load_pandas_kwargs: Optional[Dict[str, Any]] = None,
    load_pickle_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Union[tst.Dataset, List[tst.Dataset]], str]:
    """Parse input and load appropriately.

    This implements the logic which appropriately loads input and
    casts to the desired output type.

    Starting at the lowest level (file id) and moving upward in
      heirarchy through parquet and into Pandas, the dataset is
      loaded and returned in the form that you ask for.

    Order:
        * FileId -> Parquet
        * Parquet -> Dataset
        * Dataset -> Table
        * Table -> Pandas
        * Shape -> Pandas
        * Pickle -> Object

    If your particular use-case isn't accounted for, let's discuss.
    """
    output_format = output_format.lower()
    _load_funcs = {
        "parquet": lambda x: _load_dataset(x, load_dataset_kwargs),
        "dataset": lambda x: _load_dataset(x, load_dataset_kwargs),
        "table": lambda x: _load_table(x, load_table_kwargs),
        "shape": lambda x: _load_shape(x, load_pandas_kwargs),
        "pickle": lambda x: _load_pickle(x, load_pickle_kwargs),
        "pandas": lambda x: _load_pandas(x, load_pandas_kwargs),
    }
    if not isinstance(dataset_or_filepaths, List):
        _dataset_or_filepaths = [dataset_or_filepaths]
    else:
        _dataset_or_filepaths = dataset_or_filepaths
    # Now we're going to upcast until either something goes wrong
    #   or we have what we want.
    logger.debug(f"Loading: {_dataset_or_filepaths}")
    while True:
        # If the type we *have* isn't the type we want, get the
        #   next loading function and load it. This allows for
        #   iteratively mapping
        if dataset_type != output_format:
            if dataset_type == "shape":
                _dataset_or_filepaths = _load_funcs[dataset_type](_dataset_or_filepaths)
                dataset_type = "pandas"
            elif dataset_type == "pickle":
                _dataset_or_filepaths = _load_funcs[dataset_type](_dataset_or_filepaths)
                dataset_type = "object"
            else:
                try:
                    _next_type = _next_mapping[dataset_type]
                    dataset_type = _next_type
                    _dataset_or_filepaths = _load_funcs[dataset_type](
                        _dataset_or_filepaths
                    )
                except BaseException as e:
                    raise tsle.ThingStoreLoadingError(
                        f"""
                    \nUnable to load a file of type {dataset_type}.

                    Either it's not in the loading functions, or it broke in the load!

                    Loading Functions: {_load_funcs.keys()}

                    Unloadable Data
                    ---------------\n{_dataset_or_filepaths}

                    Loading Parameters
                    ------------------
                    dataset_or_filepaths: {_dataset_or_filepaths},
                    dataset_type: {dataset_type},
                    output_format: {output_format},
                    load_dataset_kwargs: {load_dataset_kwargs},
                    load_table_kwargs: {load_table_kwargs},
                    load_pandas_kwargs: {load_pandas_kwargs}
                    load_pickle_kwargs: {load_pickle_kwargs}
                    """
                    ) from e
        else:
            return _dataset_or_filepaths, dataset_type


def load(  # noqa: C901
    dataset_or_filepaths: Union[
        tst.Dataset, Iterable[tst.Dataset], Mapping[str, tst.Dataset]
    ],
    dataset_type: str,
    output_format: str = "pandas",
    load_dataset_kwargs: Optional[Dict[str, Any]] = None,
    load_table_kwargs: Optional[Dict[str, Any]] = None,
    load_pandas_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[ds.Dataset, pa.Table, pd.DataFrame]:
    """Load data in standard fashion.

    Use this loading utility to standardize the data load. This
    utility can load *parquet* documents, *shapes*, and
    *Pandas DataFrames* and it will output either as a PyArrow
    Dataset, a PyArrow Table, or a Pandas DataFrame.

    It can also load pickles, though these will be passed through
    to objects directly.

    This process can get smarter; for now it interprets the input
    to be one of:
    * An s3 filepath(s),
    * A local filepath(s),
    * A Pandas DataFrame, or
    * A File ID(s)

    If you are passing an iterable of paths they must all be on S3
    **OR** local. You may not mix and match and still expect this
    function to work appropriately. It might do something. It might
    even do what you want. Godspeed.

    For kwarg documentation please view the documentation for:

    * Pyarrow.dataset if reading parquet documents / tables.
    * Pandas if reading into Pandas.
    * GeoPandas if reading shapes (will use pandas kwargs).
    * Pickle if reading pickles.

    This can connect to, and use, a local metadata dataset or
    a remote metadata dataset.

    Parameters
    ----------
    dataset_or_filepaths: dataset
        These are the files that you wish to load.
        Could be 'file.parquet' or 'filepath.shp' or
        ['s3://bucket/prefix1', 's3://bucket/prefix2'] or
        'WhateverHappyFileID'.
    output_format: str = 'pandas'
        A choice from:
            * 'dataset' - Return the data as a PyArrow Dataset
            * 'table' - Return the data as a PyArrow Table
            * 'pandas' - Return the data as a Pandas DataFrame
    load_dataset_kwargs: Optional[Dict[str, tst.dataset_kwarg_type]] = None
        Overload the Dataset construction.
    load_table_kwargs: Optional[Dict[str, tst.table_kwarg_type]] = None
        Overload the conversion from Dataset to Table.
    load_pandas_kwargs: Optional[
    Dict[
        str, Union[tst.pandas_kwarg_type_from_table, tst.pandas_kwarg_type_from_shape]
        ]
    ] = None
        Overload the conversion from Table to Pandas DataFrame.

    Returns
    -------
    loaded_data: Union[ds.Dataset, pa.Table, pd.DataFrame]:
    """
    #################
    # Load the Data #
    #################
    dataset_or_filepaths, dataset_type = _map_load(
        dataset_or_filepaths=dataset_or_filepaths,
        dataset_type=dataset_type,
        output_format=output_format,
        load_dataset_kwargs=load_dataset_kwargs,
        load_table_kwargs=load_table_kwargs,
        load_pandas_kwargs=load_pandas_kwargs,
    )
    # At this point if we don't want pandas we are *done*.
    try:
        import geopandas as gp

        frametype = (pd.DataFrame, gp.GeoDataFrame)
    except ImportError:
        frametype = pd.DataFrame

    # If we do, we need to treat the data if requested.
    if isinstance(dataset_or_filepaths, List):  # We have a list of stuff here.
        # So... smoosh it down to one thing if we can.
        if all(isinstance(_, frametype) for _ in dataset_or_filepaths):
            return pd.concat(dataset_or_filepaths, ignore_index=True)
        elif all(isinstance(_, pa.Table) for _ in dataset_or_filepaths):
            return pa.concat_tables(dataset_or_filepaths)
        else:
            return dataset_or_filepaths
    else:
        return dataset_or_filepaths
