"""FileSystem Metadata Object.

This implements a Thing Store backed by a filesystem storage.
"""

import urllib
import importlib.util
import json
import logging
import os
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import shutil
import time
import tempfile

from datetime import datetime
from pathlib import Path
from thethingstore.file_id import parse_fileid
from thethingstore.api.error import (
    ThingStoreFileNotFoundError as TSFNFError,
    ThingStoreGeneralError as TSGError,
)
from thethingstore import thing_pointer as tp
from thethingstore.api import error as tse
from thethingstore.api.save import save as artifact_save
from thethingstore.api.load import load, materialize
from thethingstore.thing_store_base import ThingStore, register
from thethingstore._types import Dataset, FileId, Parameter, Metadata, Metric, Address
from pyarrow.fs import (
    FileSystem,
    FileSelector,
    FileType,
    copy_files,
    LocalFileSystem,
    S3FileSystem,
)
from typing import Any, Callable, Mapping, List, Optional, Union

logger = logging.getLogger(__name__)

####################################################################
#                   Supporting Functionality                       #
# ---------------------------------------------------------------- #
# This breaks out some of the code, primarily for readability.     #
####################################################################


def _make_pointer_file(
    ts: ThingStore,
    file_identifier: Union[FileId, Address],
    component: str,
    landing: str,
) -> None:
    """Write a pointer file to the local file system.

    If a simple file ID is provided, convert that into a full address.
    Check to make sure that the address points to a file that actually exists.
    Write out the address to the provided landing location in a subdirectory named by the provided component.

    Parameters
    ----------
    ts: ThingStore
        This is the data layer intended to house this pointer file.
    file_identifier: Union[FileId, Address]
        This is a file ID or a full address to create a pointer file for.
    component: str
        This is the component that this pointer is for.
    landing: str
        This is where to put the pointer file.
    """
    # If the file ID is not a full address, convert it into a local one
    if not file_identifier.startswith("fileid://"):
        file_identifier = ts._address_of(file_identifier)
    # Make sure the file exists in it's corresponding data layer
    _parsed = urllib.parse.urlparse(file_identifier)
    _file_id = _parsed.netloc
    _version = _parsed.fragment
    _ts = ts.address_to_ts(file_identifier)
    if not _ts._check_file_id(_file_id, _version):
        raise tse.ThingStoreFileNotFoundError(_file_id, _version)
    # Write out the pointer file
    f = open(os.path.join(landing, component, "ts-PTR"), "a+")
    f.write(file_identifier)
    f.close()


def recursive_make_dir(
    filesystem: FileSystem, root_path: str, *additional_prefixes: str
) -> None:
    """Recursively make directories.

    This will create directories for each of the prefixes
    provided. It will also create a root directory if it
    doesn't exist.

    This is intended to aid working around instances where write
    access to the root prefix is disabled, for instance when working
    in S3.

    Parameters
    ----------
    filesystem: FileSystem
        This is the filesystem to use.
    root_path: str
        This is the root path to create directories in.
    additional_prefixes: *str
        These are the directories to create.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> from pyarrow.fs import LocalFileSystem
    >>> with tempfile.TemporaryDirectory() as t:
    ...      recursive_make_dir(
    ...          LocalFileSystem(),
    ...          os.path.join(t, 'silly_dir'),
    ...          'subdir1',
    ...          'subdir2',
    ...      )
    ...      print(os.path.exists(f'{t}/silly_dir/subdir1/subdir2'))
    True

    """
    if not filesystem.get_file_info(root_path).type:  # Does not exist
        try:
            filesystem.create_dir(root_path, recursive=False)
        except BaseException as e:  # noqa: B036 - This is a catchall
            raise TSGError(
                f"The ThingStore was unable to create the root path: {root_path}"
            ) from e
    _current_path = root_path
    for prefix in additional_prefixes:
        _current_path = os.path.join(_current_path, prefix)
        if not filesystem.get_file_info(_current_path).type:
            filesystem.create_dir(_current_path, recursive=False)


def update_metadata_dataset(
    fs_metadata: "FileSystemThingStore",
    additional_metadata: pa.Table,
    default_timeout: float = 3.0,
) -> bool:
    """Append to a FileSystemThingStore metadata dataset.

    This appends a subset DataFrame onto maintained metadata.

    This is *not* a safe operation. If you are *at all even mildly
    uncomfortable with the idea of potentially destroying metadata*
    allow someone else to do this.

    With that said, this *should simply* append onto your existent
    dataset, *regardless of the content of your data*.

    Parameters
    ----------
    fs_metadata: FileSystemThingStore
        A Filesystem backed Thing Store.
    additional_metadata: pa.Table
        This is a table to extend the metadata.
    default_timeout: float = 3.
        This is the length of time to *wait* for an unlock.
    """
    err_msg = """Metadata Update Error:

    Updating the metadata timed out waiting for the metadata lock
    file to be updated. This file should be removed automatically
    when a metadata update is run, so this is something that will
    *require* intervention of the developers.

    Please file an issue.
    """
    start_time = datetime.now()
    # This can be tested by creating a lock prior.
    while _metadata_lock_status(fs_metadata=fs_metadata):  # type: ignore
        # While the file is locked.
        # Wait a few seconds.
        time.sleep(0.2)  # type: ignore
        current_time = datetime.now()
        if (current_time - start_time).seconds > default_timeout:
            raise TimeoutError(err_msg)
    _lock_metadata(fs_metadata=fs_metadata)
    # Go fetch the metadata fresh.
    try:
        metadata_dataset = ds.dataset(
            fs_metadata._fs_metadata_file, filesystem=fs_metadata._metadata_fs
        )
    except BaseException as e:
        _unlock_metadata(fs_metadata=fs_metadata)
        raise e
    # *IF* the schema for the metadata dataset isn't empty then we
    #   want to try and apply it to the datasets.
    if metadata_dataset.schema and metadata_dataset.count_rows():
        # We want to make a 'unified' schema which is just 'any fields that overlap
        #   are going to get the canonical type.
        existing_schema = {
            k: v
            for k, v in zip(
                metadata_dataset.schema.names, metadata_dataset.schema.types
            )
        }
        incoming_schema = {
            k: v
            for k, v in zip(
                additional_metadata.schema.names, additional_metadata.schema.types
            )
        }
        incoming_schema.update(existing_schema)
        incoming_schema = {
            k: v
            for k, v in incoming_schema.items()
            if k in additional_metadata.schema.names
        }
        # This sorts the new schema.
        try:
            additional_metadata = additional_metadata.cast(
                target_schema=pa.schema(incoming_schema)
            )
        except BaseException as e:  # noqa: B036 - This is a catchall
            raise Exception(
                f"""Schema Validation Error
            Your metadata couldn't be coerced to a standard form.

            Existing Schema
            ---------------\n{existing_schema}

            Incoming Schema
            ---------------\n{incoming_schema}

            Your Metadata
            -------------\n{additional_metadata}
            """
            ) from e

    try:
        metadata_dataset = pa.concat_tables(
            [metadata_dataset.to_table(), additional_metadata],
            promote_options="default",
        )
    except BaseException as e:  # noqa: B036 - This is a catchall
        raise Exception(metadata_dataset.schema, additional_metadata.schema) from e
    # Now write that to the Thing Store.
    pq.write_table(
        table=metadata_dataset,
        where=fs_metadata._fs_metadata_file,
        filesystem=fs_metadata._metadata_fs,
    )
    # Remove the file lock.
    _unlock_metadata(fs_metadata=fs_metadata)
    return True


def create_default_dataset(
    filesystem: FileSystem,
    path: str,
    schema: pa.Schema,
) -> None:
    """Create default, empty, file if it does not exist."""
    file_info = filesystem.get_file_info(path)
    if file_info.type.name == "NotFound":
        logger.warn(f"No file @{path}")
        logger.warn(f"Default file with schema ({schema}) created @{path}")
        _path, _ = os.path.split(path)
        if _path:
            if not isinstance(filesystem, S3FileSystem):
                filesystem.create_dir(_path)
        blank_table = pa.Table.from_pylist([], schema=schema)
        pq.write_table(
            blank_table,
            where=path,
            filesystem=filesystem,
        )


def get_user(
    metadata_path: str = "file:///opt/ml/metadata/resource-metadata.json",
) -> str:
    """Extract last string from EC2 instance name.

    By naming convention, the last word of the instance name should
    be the owner of the instance. This function extracts that string
    and returns it.

    Parameters
    ----------
    metadata_path: str = "/opt/ml/metadata/resource-metadata.json"
        The path of the metadata file in AWS.

    Returns
    -------
    user_name: str
        This should be a user alias.

    Examples
    --------
    >>> get_user()
    'USER_UNKNOWN_NO_METADATA'
    """
    try:
        fs, path = FileSystem.from_uri(metadata_path)
    except pa.lib.ArrowInvalid:  # It *might* be local!
        fs = LocalFileSystem()
        path = metadata_path
    file_status = fs.get_file_info(path)
    if file_status.type.value == 0:  # File Not Found
        return "USER_UNKNOWN_NO_METADATA"
    with fs.open_input_file(path) as json_file:
        metadata = json.load(json_file)
    instance_name = metadata["ResourceName"]
    # the last string in the list is the user alias, by convention
    user_name = instance_name.split("-")[-1]
    return user_name


def pyarrow_tree(
    path: str,
    filesystem: Union[S3FileSystem, LocalFileSystem],
    max_depth: int = 4,
    file_info: bool = True,
) -> Union[Mapping[str, Any], str]:
    """Build a tree from a URI using PyArrow filesystems.

    Parameters
    ----------
    path: str
        This is a filepath in a filesystem, i.e. 'path/to/directory'.
    filesystem: FileSystem
        This is a PyArrow FileSystem.
    max_depth: int = 4
        This is the maximum recursion depth.
    file_info: bool = True
        Return file information, or just 'file' for files.

    Returns
    -------
    tree: Mapping[str, Any]
        This is a potentially deeply nested dictionary representing
        a tree file system data structure.

    Examples
    --------
    >>> from pyarrow.fs import FileSystem
    >>> uri = 's3://path/to/directory'
    >>> # fs, path = FileSystem.from_uri(uri)  # Works with AWS access!
    >>> # pyarrow_tree(path, fs, max_depth=1)
    """
    fileset_at_prefix = filesystem.get_file_info(FileSelector(path))
    output_dict = {
        _.base_name: {
            "extension": _.extension,
            "is_file": _.is_file,
            "mtime": _.mtime,
            "mtime_ns": _.mtime_ns,
            "path": _.path,
            "size": _.size,
            "type": _.type,
        }
        for _ in fileset_at_prefix
    }
    # Short circuit this if we've gotten too deep.
    if max_depth == 0:
        return "..."
    potential_changes = {}
    for k in output_dict:
        if output_dict[k]["type"] == 3:  # Directory
            potential_changes[k] = pyarrow_tree(
                output_dict[k]["path"],
                max_depth=max_depth - 1,
                filesystem=filesystem,
                file_info=file_info,
            )
        if output_dict[k]["type"] == 2 and not file_info:  # File
            potential_changes[k] = "file"
    output_dict.update(potential_changes)
    return output_dict


####################################################################
#                         Metadata Locking                         #
####################################################################
def _lock_metadata(
    fs_metadata: "FileSystemThingStore",
) -> bool:
    """Locks metadata by writing a lockfile.

    Will raise an error if it cannot lock.

    Parameters
    ----------
    fs_metadata: FileSystemThingStore
        A Filesystem backed Thing Store.

    Returns
    -------
    is_locked: bool
        The metadata is locked.
    """
    # Get required information for lockfile location and content.
    output = pa.Table.from_pandas(
        pd.DataFrame([{"USER": get_user(), "TIME": datetime.now()}])
    )
    pq.write_table(
        output,
        where=fs_metadata._fs_metadata_lockfile,
        filesystem=fs_metadata._metadata_fs,
    )
    if not _metadata_lock_status(fs_metadata):
        raise Exception("Unable to lock metadata.")
    return True


def _unlock_metadata(
    fs_metadata: "FileSystemThingStore",
) -> bool:
    """Unlocks metadata by writing a lockfile.

    Will raise an error if it cannot unlock.

    Parameters
    ----------
    fs_metadata: FileSystemThingStore
        A Filesystem backed Thing Store.

    Returns
    -------
    is_unlocked: bool
        The metadata is locked.
    """
    pq.write_table(
        pa.Table.from_pylist([]),
        where=fs_metadata._fs_metadata_lockfile,
        filesystem=fs_metadata._metadata_fs,
    )
    if _metadata_lock_status(fs_metadata):  # pragma: no cover
        raise Exception("Unable to unlock metadata!")
    return True


def _metadata_lock_status(
    fs_metadata: "FileSystemThingStore",
) -> bool:
    """Return the lock status of the metadata file.

    Parameters
    ----------
    fs_metadata: FileSystemThingStore
        A Filesystem backed Thing Store.

    Returns
    -------
    is_metadata_locked: bool
        If this is True the metadata is locked for writing.
    """
    try:
        data = ds.dataset(
            fs_metadata._fs_metadata_lockfile, filesystem=fs_metadata._metadata_fs
        )
    except FileNotFoundError:
        return False
    # except BaseException as e:
    #     raise RuntimeError("Cannot determine lockfile status") from e
    if data.count_rows() > 0:
        return True
    else:
        return False


def convert_addr_to_pafs_kwargs(addr: Address) -> tuple[FileSystem, str]:
    """Parse an address and return kwargs.

    This function takes a full address, parses it, and returns FileSystemThingStore parameters that can be passed
    into the constructor to make a TS instance.

    If the 'AWS_ENDPOINT_URL' environment variable is set in the OS, that value will be passed into the S3FileSystem.

    Parameters
    ----------
    addr: Address
        This is an address to parse into parameters.

    Returns
    -------
    metadata_filesystem, managed_location: tuple[FileSystem, str]
        This is a tuple containing the parameters to spin up a FileSystemThingStore instance.
    """
    parsed = parse_fileid(addr)
    ###########
    # File ID #
    ###########
    _ = parsed["fileid"]  # Ignore the fileid
    ##################
    # Implementation #
    ##################
    implementation = parsed["implementation"]
    # This should the pafs implementation
    if implementation != "FileSystemThingStore":
        raise tse.ThingStoreGeneralError(
            "This is expected to be a FileSystemThingStore"
        )
    ##########
    # Params #
    ##########
    params = parsed["implementation_params"]
    fs = params["metadata_filesystem"]  # type: ignore
    managed_location = params["managed_location"]  # type: ignore
    #######################
    # Metadata Filesystem #
    #######################
    fs_map = {"LocalFileSystem": LocalFileSystem, "S3FileSystem": S3FileSystem}
    try:
        if os.getenv("AWS_ENDPOINT_URL") and fs == "S3FileSystem":
            metadata_filesystem = fs_map[fs](
                endpoint_override=os.getenv("AWS_ENDPOINT_URL")
            )
        else:
            metadata_filesystem = fs_map[fs]()
    except KeyError:
        raise NotImplementedError(
            f"""Backend filesystem is not implemented: {fs}'
                Currently implemented backend file system types: {fs_map.keys()}
                """
        )

    return metadata_filesystem, managed_location


@register
class FileSystemThingStore(ThingStore):
    """FileSystem backed ThingStore.

    For this filesystem you need to have a metadata parquet document.
    This can be initialized with an empty document that specifies
    the metadata you wish to record.

    This will grow to accept the schema you desire, but there will
    always be FILE_VERSION and FILE_ID recorded in the metadata,
    regardless of whether you provide it.
    """

    def __init__(  # noqa: C901
        self,
        metadata_filesystem: Optional[FileSystem] = None,
        managed_location: Optional[str] = None,
        address: Optional[str] = None,
    ) -> None:
        if not metadata_filesystem and not managed_location:
            if not address:
                raise tse.ThingStoreGeneralError(
                    "Provide a filesystem/location or and address"
                )
            metadata_filesystem, managed_location = convert_addr_to_pafs_kwargs(address)
        if not metadata_filesystem or not managed_location:
            raise tse.ThingStoreGeneralError(
                "A metadata filesystem or a managed location were not provided."
            )
        super().__init__(
            metadata_filesystem=metadata_filesystem, local_storage_folder=None
        )
        self._fs_metadata_file = managed_location + "/metadata.parquet"
        self._fs_metadata_lockfile = managed_location + "/metadata-lock.parquet"
        self._fs_output_location = managed_location + "/managed_files"
        if isinstance(metadata_filesystem, S3FileSystem):
            if self._fs_metadata_file.startswith("/"):
                self._fs_metadata_file = self._fs_metadata_file[1:]
            if self._fs_metadata_lockfile.startswith("/"):
                self._fs_metadata_lockfile = self._fs_metadata_lockfile[1:]
            if self._fs_output_location.startswith("/"):
                self._fs_output_location = self._fs_output_location[1:]
        self._output_location = self._fs_output_location
        # Quick checkycheck here.
        try:
            create_default_dataset(
                filesystem=metadata_filesystem,
                path=self._fs_metadata_file,
                schema=pa.schema({"FILE_ID": "str", "FILE_VERSION": "int64"}),
            )
        except BaseException as e:  # noqa: B036 - Catchall
            raise TSGError("Unable to create metadata file.") from e
        try:
            create_default_dataset(
                filesystem=metadata_filesystem,
                path=self._fs_metadata_lockfile,
                schema=pa.schema({"USER": "str"}),
            )
        except BaseException as e:  # noqa: B036 - Catchall
            raise TSGError("Unable to create metadata lockfile.") from e

    def _delete(self, file_id: FileId) -> None:
        """Delete a file.

        Calling delete upon any individual file will accomplish one of two things, depending on the current state:
        * If the file is valid `(DATASET_VALID / THING_VALID == True)` this will
        log a new version of the file, with metadata only, and with DATASET_VALID set to False.
        * If the file is not valid `(DATASET_VALID / THING_VALID == False)` this
        will destructively remove that file (both metadata and contents.)

        Parameters
        ----------
        file_id: FileID
            This is the identifier for the file in the ThingStore.
        """
        # Grab TS connection info and parquet file
        _filesystem = self._metadata_fs
        metadata_parquet_path = self._fs_metadata_file
        metadata_folder_path = self._fs_output_location
        df = self.browse()
        # Checks if the file is valid
        try:
            _metadata = dict(self.get_metadata(file_id))
        except Exception as e:
            raise KeyError(
                f" Unable to delete because {file_id} does not exist!"
            ) from e
        if _metadata["DATASET_VALID"]:
            _metadata["DATASET_VALID"] = "FALSE"
            self.log(metadata=_metadata)
            df = self.browse()
        elif not _metadata["DATASET_VALID"]:
            # Make a backup before anything destructive
            bkp_path = metadata_parquet_path.replace(
                ".parquet", f'{datetime.now().strftime("%Y%m%d%H%M")}.bkp'
            )
            pq.write_table(
                ds.dataset(metadata_parquet_path, filesystem=_filesystem).to_table(),
                bkp_path,
                filesystem=_filesystem,
            )
            if not _filesystem.get_file_info(bkp_path):
                raise Exception("Backup was not found")
            df = df.drop(df[(df["FILE_ID"] == file_id)].index)
            delete_path = os.path.join(metadata_folder_path, file_id)
            _filesystem.delete_dir(delete_path)
            logger.warning(f"Deleted {file_id}")
        if isinstance(_filesystem, S3FileSystem):
            with tempfile.TemporaryDirectory() as t:
                # Pandas is not playing nicely with moto, so we write to local first, then copy to s3 with pyarrow
                temp_path = os.path.join(f"{t}/data.parquet")
                df.to_parquet(path=temp_path)
                copy_files(
                    source=temp_path,
                    destination=metadata_parquet_path,
                    source_filesystem=LocalFileSystem(),
                    destination_filesystem=self._metadata_fs,
                )
        else:
            df.to_parquet(path=metadata_parquet_path, engine="pyarrow")

    def _update(self, file_id: FileId, **kwargs: Any) -> None:
        """Update a file.

        Update is a convenience function which logs a copy of the most recent version
        of a FILE_ID with specified components updated.
        Any components not specified will be logged as pointers to the latest version.

        Parameters
        ----------
        file_id: FileID
            This is the identifier for the file in the ThingStore.
        **kwargs
            Specific components of the file that are passed into _update
        """
        if "metadata" not in kwargs:
            kwargs["metadata"] = {"FILE_ID": file_id}
        elif "FILE_ID" not in kwargs["metadata"]:
            kwargs["metadata"].update({"FILE_ID": file_id})
        elif kwargs["metadata"]["FILE_ID"] != file_id:
            raise tse.ThingStoreNotAllowedError(
                "Update a file ID's file ID", "You cannot change a file's ID."
            )

        # Log a new version of the file ID
        # Let log handle creating pointers to any components that aren't overridden
        # NOTE: It is impossible to create a cycle as every update creates a new version
        #   with no preexisting dependencies.
        self.log(**kwargs)

    def _load(
        self,
        file_identifier: FileId,
        version: Optional[str] = None,
        component: str = "data",
        **kwargs: Any,
    ) -> Union[Dataset, None]:
        """Convert a FileId to dataset.

        This returns a representation of the dataset (Dataset)
        or None if no dataset exists.

        Parameters
        ----------
        file_identifier: FileId
            This is a file identifier understood by the Thing Store.

        Returns
        -------
        output_data: Union[Dataset, None]
            Output data in the form requested.
        """
        # This blows up if the file ID DNE!
        _ = self._get_metadata(file_identifier=file_identifier, version=version)
        dataset_location = os.path.join(
            self._output_location, file_identifier, str(_["FILE_VERSION"]), component
        )
        if not version:
            version = str(_["FILE_VERSION"])
        if self._is_pointer(file_identifier, component, version=version):
            return tp.dereference(self, file_identifier, component, version=version)
        if self._metadata_fs.get_file_info(dataset_location).type == 0:  # Not found
            return None
        else:
            return ds.dataset(dataset_location, filesystem=self._metadata_fs, **kwargs)

    def _log(  # noqa: C901
        self,
        dataset: Optional[Union[Dataset, FileId, Address]] = None,
        parameters: Optional[Union[Mapping[str, Parameter], FileId, Address]] = None,
        metadata: Optional[Union[Mapping[str, Optional[Metadata]], FileId]] = None,  # type: ignore
        metrics: Optional[Union[Mapping[str, Metric], FileId, Address]] = None,  # type: ignore
        artifacts_folder: Optional[Union[str, FileId, Address]] = None,
        embedding: Optional[Union[Dataset, FileId]] = None,
    ) -> FileId:
        """Store a file in the Thing Store and associated information in the metadata.

        This will load your information into the Thing Store.

        For certain components, a File ID or a full address can be provided in place of the actual data.
        In this case, a reference to that data will be logged instead of the actual data. This is allowed for
        the following components:
            * dataset
            * parameters
            * metrics
            * artifacts

        Reference files (ThingPointers) will be created under the following conditions
            * A file ID is provided in place of one of the allowed component types.
                * A pointer file will be logged in place of the actual data.
                    * If a simple file ID is provided, this will be treated as being local
                    to this data layer (i.e. self)
                    * Else if a full address is provided, this will be copied directly into the pointer file.
            * The file ID value of the metadata already exists in the data layer.
                * This will be treated as a version update.
                * For all allowed components, if that component is not provided, log a pointer file to the
                latest version.

        Parameters
        ----------
        dataset: Optional[Union[Dataset, FileId, Address]] = None
            This is understandable as a dataset. This can also be a
        parameters: Optional[Mapping[str, Parameter]] = None
            These are parameters that may be logged with this file.
        metadata: Optional[Mapping[str, Optional[Metadata]]] = None
            These are additional metadata elements that may be
            logged with this file.
        metrics: Optional[Mapping[str, Metric]] = None
            These are additional metrics that may be logged with
            this file.
        artifacts_folder: Optional[str] = None
            This is a folderpath that may be collected and logged with this file.
        embedding: Optional[Dataset] = None
            This is a (set of) two-dimensional embedding(s).

        Returns
        -------
        file_identifier: FileId
            This is a file identifier understood by the Thing Store.
            If this is not unique this will raise an exception.
        """
        if metadata is None:
            _metadata: Mapping[str, Optional[Metadata]] = {}  # type: ignore
        else:
            _metadata: Mapping[str, Optional[Metadata]] = metadata  # type: ignore

        _metadata = self._scrape_metadata(_metadata)
        # Are we making an update to an already existing file?
        _update_version = self._check_file_id(str(_metadata["FILE_ID"]))
        output_path = os.path.join(
            self._output_location,
            str(_metadata["FILE_ID"]),
            str(_metadata["FILE_VERSION"]),
        )
        recursive_make_dir(
            self._metadata_fs,
            self._output_location,
            str(_metadata["FILE_ID"]),
            str(_metadata["FILE_VERSION"]),
        )
        with tempfile.TemporaryDirectory() as t:
            # 1. Save the dataset {file_id}/data
            if dataset is not None:
                os.makedirs(f"{t}/data/")
                # If the dataset is a string and is either an address or a file ID in the local TS, make a pointer
                if isinstance(dataset, str) and (
                    dataset.startswith("fileid://") or self._check_file_id(dataset)
                ):
                    _make_pointer_file(self, dataset, "data", t)
                else:
                    artifact_save(dataset, f"{t}/data/data")
            elif _update_version and _metadata["TS_HAS_DATASET"]:
                os.makedirs(f"{t}/data/")
                _ = self.get_metadata(str(_metadata["FILE_ID"]))
                _addr = self._address_of(
                    str(_metadata["FILE_ID"]),
                    version=str(_["FILE_VERSION"]),
                )
                _make_pointer_file(self, _addr, "data", t)
            # 2. Save the parameters {file_id}/parameters
            if parameters is not None:
                os.makedirs(f"{t}/parameters/")
                # If 'parameters' is a string and is either an address or a file ID in the local TS, make a pointer
                if isinstance(parameters, str) and (
                    parameters.startswith("fileid://")
                    or self._check_file_id(parameters)
                ):
                    _make_pointer_file(self, parameters, "parameters", t)
                else:
                    artifact_save(parameters, f"{t}/parameters/params")
            elif _update_version and _metadata["TS_HAS_PARAMETERS"]:
                os.makedirs(f"{t}/parameters/")
                _ = self.get_metadata(str(_metadata["FILE_ID"]))
                _addr = self._address_of(
                    str(_metadata["FILE_ID"]),
                    version=str(_["FILE_VERSION"]),
                )
                _make_pointer_file(self, _addr, "parameters", t)
            # 3. Save the metadata into a job-specific and master dataset
            if isinstance(metadata, str):
                tse.ThingStoreNotAllowedError(
                    "Creating a pointer to metadata is not allowed."
                )
            if _metadata is not None:
                os.makedirs(f"{t}/metadata")
                _metadata_to_add = pa.Table.from_pandas(
                    pd.DataFrame([_metadata]),
                    schema=pa.Schema.from_pandas(pd.DataFrame([_metadata])),
                )
                # aaaand save it here, too.
                artifact_save(_metadata_to_add, f"{t}/metadata/metadata")
            # 4. Save the metrics {file_id}/metrics
            if metrics is not None:
                os.makedirs(f"{t}/metrics")
                # If 'metrics' is a string and is either an address or a file ID in the local TS, make a pointer
                if isinstance(metrics, str) and (
                    metrics.startswith("fileid://") or self._check_file_id(metrics)
                ):
                    _make_pointer_file(self, metrics, "metrics", t)
                else:
                    artifact_save(metrics, f"{t}/metrics/metrics")
            elif _update_version and _metadata["TS_HAS_METRICS"]:
                os.makedirs(f"{t}/metrics")
                _ = self.get_metadata(str(_metadata["FILE_ID"]))
                _addr = self._address_of(
                    str(_metadata["FILE_ID"]),
                    version=str(_["FILE_VERSION"]),
                )
                _make_pointer_file(self, _addr, "metrics", t)
            # 5. Save the artifacts
            if artifacts_folder is not None:
                if self._check_file_id(artifacts_folder) or artifacts_folder.startswith(
                    "fileid://"
                ):
                    os.makedirs(f"{t}/artifacts")
                    _make_pointer_file(self, artifacts_folder, "artifacts", t)
                else:
                    shutil.copytree(artifacts_folder, f"{t}/artifacts")
            elif _update_version and _metadata["TS_HAS_ARTIFACTS"]:
                os.makedirs(f"{t}/artifacts")
                _ = self.get_metadata(str(_metadata["FILE_ID"]))
                _addr = self._address_of(
                    str(_metadata["FILE_ID"]),
                    version=str(_["FILE_VERSION"]),
                )
                _make_pointer_file(self, _addr, "artifacts", t)
            # 6. Save the embedding
            if embedding is not None:
                os.makedirs(f"{t}/embedding/")
                artifact_save(embedding, f"{t}/embedding/embedding")
            # 7. Create the necessary directories in the data layer
            output_path = os.path.join(
                self._output_location,
                str(_metadata["FILE_ID"]),
                str(_metadata["FILE_VERSION"]),
            )
            if isinstance(self._metadata_fs, S3FileSystem):
                # Cheat.
                self._metadata_fs.create_dir(output_path, recursive=True)
                pq.write_table(
                    table=pa.Table.from_pylist([]),
                    where=os.path.join(output_path, "temp"),
                    filesystem=self._metadata_fs,
                )
                self._metadata_fs.delete_file(os.path.join(output_path, "temp"))
            else:
                self._metadata_fs.create_dir(output_path)
            # 8. Use filesystems to just copy stuff over!
            if isinstance(self._metadata_fs, S3FileSystem):
                # TODO I think we want some documentation on what, exactly, this is doing.
                for root, _, files in os.walk(t):
                    dir_path = "/".join(root.split("/")[3:])
                    if dir_path:
                        dir_path = dir_path + "/"
                    for f in files:
                        copy_files(
                            source=os.path.join(root, f),
                            destination=output_path + "/" + dir_path + f,
                            source_filesystem=LocalFileSystem(),
                            destination_filesystem=self._metadata_fs,
                        )
            else:
                copy_files(
                    source=t,
                    destination=output_path,
                    source_filesystem=LocalFileSystem(),
                    destination_filesystem=self._metadata_fs,
                )
        file_id = _metadata["FILE_ID"]
        if not isinstance(file_id, FileId):
            raise Exception
        if metadata is not None:
            update_metadata_dataset(
                fs_metadata=self, additional_metadata=_metadata_to_add
            )
        return file_id

    def _address_of(
        self, file_identifier: FileId, version: Optional[str] = None
    ) -> str:
        """Build an address from file ID and version information.

        This function constructs a FileSystemThingStore address based on the following information:
            * The provided file ID.
            * The current TS object class name.
            * The current TS filesystem class name.
            * The current TS managed_location.
            * The provided version number.
                * If a version number is not provided, this is left blank.

        Parameters
        ----------
        file_identifier: FileId
            A file ID to convert into an address.
        version: Optional[str]
            An optional file version to add to the address.

        Returns
        -------
        addr: str
            The generated address.
        """
        _addr = "fileid://"
        _addr += f"{file_identifier}/"
        _addr += f"{self.__class__.__name__}?"
        _addr += f"metadata_filesystem={self._metadata_fs.__class__.__name__}&"
        _addr += f"managed_location={os.path.split(self._fs_output_location)[0]}"
        if version:
            _addr += f"#{version}"
        return _addr

    def _list_artifacts(
        self, file_identifier: FileId, version: Optional[str] = None
    ) -> List[str]:
        # Check that latest run works.
        # This blows up if the file ID DNE!
        _ = self._get_metadata(file_identifier=file_identifier, version=version)
        artifacts_path = os.path.join(
            self._output_location,
            file_identifier,
            str(_["FILE_VERSION"]),
            "artifacts",
            # Networking pathname accommodation.
        ).replace("\\", "/")
        try:
            artifacts = self._metadata_fs.get_file_info(
                FileSelector(artifacts_path, recursive=True)
            )
        except FileNotFoundError:
            logger.warn(f"No artifacts for FILE_ID@{file_identifier}")
            return []

        artifacts = pd.DataFrame(
            [
                {
                    "base_name": _.base_name,
                    "extension": _.extension,
                    "is_file": _.is_file,
                    "mtime": _.mtime,
                    "mtime_ns": _.mtime_ns,
                    "path": _.path,
                    "size": _.size,
                    "type": _.type,
                }
                for _ in artifacts
                if not _.path.endswith("artifacts") and _.is_file
            ]
        )
        artifacts = artifacts.path.str.replace(artifacts_path, "")
        mask = artifacts.str.startswith("/")
        artifacts = artifacts[mask].str.slice(1)
        return artifacts.to_list()

    def _is_subpath(self, sub_path: str, of_paths: Union[str, List[str]]) -> bool:
        """Check if a path is a subpath of one or more other paths.

        This performs substring matching, not actual filesystem analysis.
        However, because paths are converted to absolute paths, this should be relatively accurate.

        Parameters
        ----------
        sub_path: str
            This is the potential subpath
        of_paths: Union[str, List[str]]
            This is a path, or list of paths, to compare sub_path against.
        """
        # If of_paths is not a list, make it one.
        if isinstance(of_paths, str):
            of_paths = [of_paths]
        # Turn the list of paths into a list of absolute paths
        abs_of_paths = [os.path.abspath(of_path) for of_path in of_paths]
        # Returns True if the absolute subpath is a substring of any of the absolute paths
        # Returns False if not
        return any(os.path.abspath(sub_path) in abs_path for abs_path in abs_of_paths)

    def _get_artifact(
        self,
        file_identifier: FileId,
        artifact_identifier: str,
        target_path: str,
        version: Optional[str] = None,
    ) -> None:
        """Copy an artifact locally.

        This copies an artifact from the thing store to a local path.

        The artifact can be a path to a file or a directory:
        - If the path leads to a file, that file will be copied into the target directory without
          its upper directory structure.
        - If the path leads to a directory, that directory (and subordinate structure) will be recursively copied
          into the target directory without its upper directory structure.
          - An empty string functions as the root directory and will result in all artifacts being copied.

        Parameters
        ----------
        file_identifier: FileId
            This is a file identifier understood by the Thing Store.
        artifact_identifer: str
            This is the string key for the artifact (filepath in the folder).
            An emtpy string signifies the root (all artifacts).
        target_path: str
            This is where you wish to move the file.
        version: Optional[str]
            This is an optional file version. If this is not provided, get the latest version.
        """
        # The artifact ID must be:
        # 1. an empty string. This is the root (i.e. get all artifacts).
        # 2. a subpath of an element in the artifact list
        _artifact_list = self._list_artifacts(file_identifier, version)
        if artifact_identifier and not self._is_subpath(
            artifact_identifier, _artifact_list
        ):
            raise KeyError(
                f"""Unable to find artifact '{artifact_identifier}' because it does not exist in artifact list """
                + str(_artifact_list)
            )

        _ = self._get_metadata(file_identifier=file_identifier, version=version)
        artifact_path = str(
            Path(self._output_location)
            / file_identifier
            / str(_["FILE_VERSION"])
            / "artifacts"
            / artifact_identifier
        ).replace("\\", "/")
        # Does the target path exist?
        tgt_exists = os.path.exists(target_path)
        if tgt_exists:
            tgt_is_dir = os.path.isdir(target_path)
        else:
            tgt_is_dir = False
        if tgt_exists and not tgt_is_dir:
            raise Exception("CANNOT OVERWRITE FILES!")
        # Make target directories. If the artifact itself is a directory, make that one too
        _flattened_target = str(
            Path(target_path) / os.path.split(artifact_identifier)[1]
        )
        if os.path.isdir(artifact_path):
            os.makedirs(
                str(Path(target_path) / os.path.split(artifact_identifier)[1]),
                exist_ok=True,
            )
        copy_files(
            source=artifact_path,
            source_filesystem=self._metadata_fs,
            destination=_flattened_target,
            destination_filesystem=LocalFileSystem(),
        )

    def _get_parameters(
        self,
        file_identifier: FileId,
        version: Optional[str] = None,
        filesystem: Optional[FileSystem] = None,
    ) -> Mapping[str, Parameter]:
        _ = self._get_metadata(file_identifier=file_identifier, version=version)
        parameter_location = os.path.join(
            self._output_location, file_identifier, str(_["FILE_VERSION"]), "parameters"
        )
        if filesystem is None:
            filesystem = self._metadata_fs
        # Each of the parameters is logged (should be as single file).
        # Here if they exist they are unpacked and converted into a known format.
        try:
            ls_dir = [
                _.base_name
                for _ in self._metadata_fs.get_file_info(
                    FileSelector(parameter_location, recursive=True)
                )
            ]
        except FileNotFoundError:
            logger.warn(f"No parameters for FILE_ID@{file_identifier}")
            return {}
        if len(ls_dir) == 1:
            parameters = materialize(
                os.path.join(parameter_location, ls_dir[0]), filesystem=filesystem
            )
        else:
            parameters = materialize(parameter_location, filesystem=filesystem)
        if isinstance(parameters, list) and len(parameters) == 1:
            parameters = parameters[0]
        return parameters

    def _get_metadata(
        self, file_identifier: FileId, version: Optional[str] = None
    ) -> Mapping[str, Metadata]:  # type: ignore
        # Read from the metadata parquet document.
        if version:
            if not self._check_file_id(file_identifier, version=version):
                raise TSFNFError(file_identifier=file_identifier, version=version)
        if not self._check_file_id(file_identifier, version=version):
            raise TSFNFError(file_identifier=file_identifier)

        if version:
            metadata_dataset = self._browse(
                table_kwargs={
                    "filter": (pc.field("FILE_ID") == file_identifier)
                    & (pc.field("FILE_VERSION") == int(version))
                }
            )
        else:
            metadata_dataset = self._browse(
                table_kwargs={"filter": pc.field("FILE_ID") == file_identifier}
            )
        if metadata_dataset.empty:
            logger.warn(
                f"No metadata exists for {file_identifier}. Generating default."
            )
            metadata_dataset = pd.DataFrame(
                [
                    {
                        "FILE_ID": file_identifier,
                        "DATASET_DATE": "None",
                        "DATASET_VALID": "True",
                        "FILE_VERSION": 1,
                    }
                ]
            )
        # Now that I've got the data...
        if "FILE_VERSION" in metadata_dataset:  # If I have FILE_VERSION
            # Grab the latest file version.
            latest_run = metadata_dataset.loc[
                metadata_dataset[["FILE_VERSION"]].idxmax().item()
            ]
        else:  # If I *don't*
            # Just grab the latest record by index.
            latest_run = metadata_dataset.loc[metadata_dataset.index.max()]
        output = latest_run.to_dict()
        if not output:
            logger.warn(f"No metadata for FILE_ID@{file_identifier}")
            return {}
        return output

    def _get_metrics(
        self,
        file_identifier: FileId,
        version: Optional[str] = None,
        filesystem: Optional[FileSystem] = None,
    ) -> Mapping[str, Metric]:  # type: ignore
        _ = self._get_metadata(file_identifier=file_identifier, version=version)
        metric_location = os.path.join(
            self._output_location, file_identifier, str(_["FILE_VERSION"]), "metrics"
        )
        if filesystem is None:
            filesystem = self._metadata_fs
        try:
            _ = [
                _.base_name
                for _ in self._metadata_fs.get_file_info(
                    FileSelector(metric_location, recursive=True)
                )
            ]
        except FileNotFoundError:
            logger.warn(f"No metrics for FILE_ID@{file_identifier}")
            return {}
        # Here if they exist they are unpacked and converted into a known format.
        if len(_) == 1:
            metrics = materialize(
                os.path.join(metric_location, _[0]), filesystem=filesystem
            )
        else:
            metrics = materialize(metric_location, filesystem=filesystem)
        if isinstance(metrics, list) and len(metrics) == 1:
            metrics = metrics[0]
        return metrics

    def _get_function(
        self,
        file_identifier: FileId,
        version: Optional[str] = None,
        filesystem: Optional[FileSystem] = None,
    ) -> Optional[Callable]:
        _ = self._get_metadata(file_identifier=file_identifier, version=version)
        function_location = os.path.join(
            self._output_location,
            file_identifier,
            str(_["FILE_VERSION"]),
            "artifacts/function",
        )
        if filesystem is None:
            filesystem = self._metadata_fs
        try:
            _ = [
                _.base_name
                for _ in self._metadata_fs.get_file_info(
                    FileSelector(function_location, recursive=True)
                )
            ]
        except FileNotFoundError:
            logger.warn(f"No function for FILE_ID@{file_identifier}")
            return None
        with tempfile.TemporaryDirectory() as t:
            # This uses standard Python import libraries
            #   to bring down a Python file and read it into
            #   scope. After this it's available as a module.
            py_file_path = f"{t}/workflow.py"
            self.get_artifact(file_identifier, "function/workflow.py", t)
            _, py_file_name = os.path.split(py_file_path)
            spec = importlib.util.spec_from_file_location(
                py_file_name.replace(".py", ""), py_file_path.replace(".py", "") + ".py"
            )
            assert spec is not None  # nosec
            _module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_module)  # type: ignore
            return _module.workflow

    def _browse(self, **kwargs: Any) -> pd.DataFrame:
        if kwargs is None:
            kwargs = {}
        dataset_kwargs = kwargs.get("dataset_kwargs", {})
        if dataset_kwargs is None:
            dataset_kwargs = {}
        if "filesystem" not in dataset_kwargs:
            dataset_kwargs.update(filesystem=self._metadata_fs)
        table_kwargs = kwargs.get("table_kwargs", {})
        if table_kwargs is None:
            table_kwargs = {}
        pandas_kwargs = kwargs.get("pandas_kwargs", {})
        if pandas_kwargs is None:
            pandas_kwargs = {}
        metadata_dataset = (
            ds.dataset(self._fs_metadata_file, **dataset_kwargs)
            .to_table(**table_kwargs)
            .to_pandas(**pandas_kwargs)
            .reset_index(drop=True)
        )
        return metadata_dataset

    def _get_embedding(
        self, file_identifier: FileId, version: Optional[str] = None, **kwargs: Any
    ) -> ds.Dataset:
        try:
            import ibis
        except ImportError:
            raise Exception("Please run `pip install ibis` to use embeddings.")
        try:
            import torch
        except ImportError:
            raise Exception("Please run `pip install torch` to use embeddings.")
        _params: dict[str, Any] = {"dataset_type": "dataset", "output_format": "table"}
        _params.update(kwargs)

        torch_map = ibis.memtable(
            load(
                self._load(
                    file_identifier=file_identifier,
                    component="embedding",
                    version=version,
                ),
                **_params,
            )
        ).to_torch()

        return torch.stack(tuple(torch_map.values()), dim=1)

    def _get_pointer(
        self, file_identifier: FileId, component: str, version: Optional[str] = None
    ) -> str:
        """Read the value stored in a pointer.

        Given the necessary information to access a pointer file, read and return the contents.

        Parameters
        ----------
        file_identifier: FileId
            This is the file ID containing the pointer.
        component: str
            This is the component containing the pointer.
        version: Optional[str] = None
            This is the file version where the pointer is located. If not provided, this will be treated as latest.

        Returns
        -------
        pointer: str
            The value stored in the pointer file.
        """
        if not self._check_file_id(file_identifier, version=version):
            raise tse.ThingStoreFileNotFoundError(file_identifier, version=version)
        if not self._is_pointer(file_identifier, component, version):
            raise tse.ThingStorePointerError(
                file_identifier, component, "This file is not a pointer"
            )

        _ = self._get_metadata(file_identifier, version=version)
        _filepath = os.path.join(
            self._fs_output_location,
            file_identifier,
            str(_["FILE_VERSION"]),
            component,
            "ts-PTR",
        )
        with self._metadata_fs.open_input_file(_filepath) as f:
            return f.read().decode()

    def _post_process_browse(self, browse_results: pd.DataFrame) -> pd.DataFrame:
        return browse_results

    def _check_file_id(
        self, file_identifier: FileId, version: Optional[str] = None
    ) -> bool:
        """Determine if a file id is in the metadata."""
        # convert a full address to a local file ID
        if file_identifier.startswith("fileid://"):
            _parsed = urllib.parse.urlparse(file_identifier)
            file_identifier = _parsed.netloc
            version = _parsed.fragment
        # File IDs don't start with a slash
        if file_identifier.startswith("/"):
            return False
        # Go get the root path for the file id.
        _flpath = os.path.join(self._fs_output_location, file_identifier)
        if version:
            _flpath = os.path.join(_flpath, version)
        fl_info = self._metadata_fs.get_file_info(_flpath)
        # 'Not' not found.
        return not fl_info.type == 0  # This is a File Not Found code.

    def _is_pointer(
        self, file_identifier: FileId, component: str, version: Optional[str] = None
    ) -> bool:
        """Determine if this is a pointer."""
        _ = self._get_metadata(file_identifier, version=version)
        _filepath = os.path.join(
            self._fs_output_location,
            file_identifier,
            str(_["FILE_VERSION"]),
            component,
            "ts-PTR",
        )
        fl_info = self._metadata_fs.get_file_info(_filepath)
        # Same logic as _check_file_id
        return not fl_info.type == 0

    def _test_field_value(self, field: str, value: Metadata) -> bool:  # type: ignore
        _data = ds.dataset(
            self._fs_metadata_file,
            filesystem=self._metadata_fs,
        )
        if field not in _data.schema.names:
            return False
        metadata_dataset = _data.to_table(filter=pc.field(field) == value).to_pandas()
        if not metadata_dataset.empty:
            return True
        else:
            return False

    def _get_size(self, file_identifier: FileId, version: Optional[str] = None) -> int:
        """Return the size of the file ID directory.

        If a version is given, only return the size of that version.

        Parameters
        ----------
        file_identifier: FileId
            The file ID for which to get the size.
        version:
            An optional file version to the file ID.

        Returns
        -------
        total_size: int
            The total size, in bytes, of the file ID.
        """
        if not self._check_file_id(file_identifier, version=version):
            raise tse.ThingStoreFileNotFoundError(file_identifier, version=version)
        # Get path to the file ID directory
        filepath = os.path.join(self._fs_output_location, file_identifier)
        if version:
            filepath = os.path.join(filepath, version)
        # Calculate the cumulative size of the files
        total_size = 0
        selector = FileSelector(filepath, recursive=True)
        for file_info in self._metadata_fs.get_file_info(selector):
            if file_info.type == FileType.File:
                total_size += file_info.size
        return total_size
