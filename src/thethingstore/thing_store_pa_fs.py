"""FileSystem Metadata Object.

This implements a Thing Store backed by an S3 storage.
This requires specifying a bucket and a few prefixes.
"""
import json
import logging
import os
import tempfile
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import shutil
import time
from datetime import datetime
from thethingstore.api.error import ThingStoreFileNotFoundError as TSFNFError
from thethingstore.api.save import save as artifact_save
from thethingstore.api.load import materialize
from thethingstore.thing_store_base import ThingStore
from thethingstore.types import Dataset, FileId, Parameter, Metadata, Metric
from pyarrow.fs import (
    FileSystem,
    FileSelector,
    copy_files,
    LocalFileSystem,
    S3FileSystem,
)
from typing import Any, Mapping, List, Optional, Union

logger = logging.getLogger(__name__)

####################################################################
#                   Supporting Functionality                       #
# ---------------------------------------------------------------- #
# This breaks out some of the code, primarily for readability.     #
####################################################################


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
    while _metadata_lock_status(fs_metadata=fs_metadata):
        # While the file is locked.
        # Wait a few seconds.
        time.sleep(0.2)
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
        except BaseException as e:
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
            [metadata_dataset.to_table(), additional_metadata], promote=True
        )
    except BaseException as e:
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


class FileSystemThingStore(ThingStore):
    """This is a FileSystem backed API.

    For this filesystem you need to have a metadata parquet document.
    This can be initialized with an empty document that specifies
    the metadata you wish to record.

    This will grow to accept the schema you desire, but there will
    always be FILE_VERSION and FILE_ID recorded in the metadata,
    regardless of whether you provide it.
    """

    def __init__(
        self,
        metadata_filesystem: FileSystem,
        managed_location: str,
    ) -> None:
        super().__init__(
            metadata_filesystem=metadata_filesystem, local_storage_folder=None
        )
        self._fs_metadata_file = managed_location + "/metadata.parquet"
        self._fs_metadata_lockfile = managed_location + "/metadata-lock.parquet"
        self._fs_output_location = managed_location + "/managed_files"
        self._output_location = self._fs_output_location
        if isinstance(metadata_filesystem, S3FileSystem):
            if self._fs_metadata_file.startswith("/"):
                self._fs_metadata_file = self._fs_metadata_file[1:]
            if self._fs_metadata_lockfile.startswith("/"):
                self._fs_metadata_lockfile = self._fs_metadata_lockfile[1:]
            if self._fs_output_location.startswith("/"):
                self._fs_output_location = self._fs_output_location[1:]
        # Quick checkycheck here.
        create_default_dataset(
            filesystem=metadata_filesystem,
            path=self._fs_metadata_file,
            schema=pa.schema({"FILE_ID": "str", "FILE_VERSION": "int64"}),
        )
        create_default_dataset(
            filesystem=metadata_filesystem,
            path=self._fs_metadata_lockfile,
            schema=pa.schema({"USER": "str"}),
        )

    def _load(self, file_identifier: FileId, **kwargs) -> Union[Dataset, None]:
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
        _ = self._get_metadata(file_identifier=file_identifier)
        dataset_location = os.path.join(
            self._output_location, file_identifier, str(_["FILE_VERSION"]), "data"
        )
        if self._metadata_fs.get_file_info(dataset_location).type == 0:  # Not found
            return None
        else:
            return ds.dataset(dataset_location, filesystem=self._metadata_fs, **kwargs)

    def _log(  # noqa: C901
        self,
        dataset: Optional[Dataset] = None,
        parameters: Optional[Mapping[str, Parameter]] = None,
        metadata: Optional[Mapping[str, Optional[Metadata]]] = None,
        metrics: Optional[Mapping[str, Metric]] = None,
        artifacts_folder: Optional[str] = None,
    ) -> FileId:
        """Store a file in the Thing Store and associated information in the metadata.

        This will load your information into the Thing Store.

        Parameters
        ----------
        dataset: Optional[Dataset] = None
            This is understandable as a dataset.
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

        Returns
        -------
        file_identifier: FileId
            This is a file identifier understood by the Thing Store.
            If this is not unique this will raise an exception.
        """
        if metadata is None:
            _metadata: Mapping[str, Optional[Metadata]] = {}
        else:
            _metadata = metadata

        _metadata = self._scrape_metadata(_metadata)
        output_path = os.path.join(
            self._output_location,
            str(_metadata["FILE_ID"]),
            str(_metadata["FILE_VERSION"]),
        )
        if isinstance(self._metadata_fs, S3FileSystem):
            # Cheat.
            pq.write_table(
                table=pa.Table.from_pylist([]),
                where=os.path.join(output_path, "temp"),
                filesystem=self._metadata_fs,
            )
            self._metadata_fs.delete_file(os.path.join(output_path, "temp"))
        else:
            self._metadata_fs.create_dir(output_path)
        with tempfile.TemporaryDirectory() as t:
            # 1. Save the dataset {file_id}/data
            if dataset is not None:
                os.makedirs(f"{t}/data/")
                artifact_save(dataset, f"{t}/data/data")
            # 2. Save the parameters {file_id}/parameters
            if parameters is not None:
                os.makedirs(f"{t}/parameters/")
                artifact_save(parameters, f"{t}/parameters/params")
            # 3. Save the metadata into a job-specific and master dataset
            if _metadata is not None:
                os.makedirs(f"{t}/metadata")
                _metadata_to_add = pa.Table.from_pandas(
                    pd.DataFrame([_metadata]),
                    schema=pa.Schema.from_pandas(pd.DataFrame([_metadata])),
                )
                artifact_save(_metadata_to_add, f"{t}/metadata/metadata")
            # 4. Save the metrics {file_id}/metrics
            if metrics is not None:
                os.makedirs(f"{t}/metrics")
                artifact_save(metrics, f"{t}/metrics/metrics")
            # 5. Save the artifacts
            if artifacts_folder is not None:
                shutil.copytree(artifacts_folder, f"{t}/artifacts")
            # 6. Use filesystems to just copy stuff over!
            if isinstance(self._metadata_fs, S3FileSystem):
                # TODO I think we want some documentation on what, exactly, this is doing.
                for root, dirs, files in os.walk(t):
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

    def _list_artifacts(self, file_identifier: FileId) -> List[str]:
        # Check that latest run works.
        # This blows up if the file ID DNE!
        _ = self._get_metadata(file_identifier=file_identifier)
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

    def _get_artifact(
        self, file_identifier: FileId, artifact_identifier: str, target_path: str
    ) -> None:
        """Copy an artifact locally.

        This copies an artifact from the thing store to a local path.
        This will append 'artifacts' to the path you provide.

        Parameters
        ----------
        file_identifier: FileId
            This is a file identifier understood by the Thing Store.
        artifact_identifer: str
            This is the string key for the artifact (filepath in the folder).
        target_path: str
            This is where you wish to move the file.
        """
        _ = self._get_metadata(file_identifier=file_identifier)
        artifact_path = os.path.join(
            self._output_location,
            file_identifier,
            str(_["FILE_VERSION"]),
            "artifacts",
            artifact_identifier,
        ).replace("\\", "/")
        _, flnm = os.path.split(artifact_path)
        # 1. Does the target path exist?
        tgt_exists = os.path.exists(target_path)
        if tgt_exists:
            tgt_is_dir = os.path.isdir(target_path)
        else:
            tgt_is_dir = False
        if tgt_exists and not tgt_is_dir:
            raise Exception("CANNOT OVERWRITE FILES!")
        os.makedirs(os.path.join(target_path, "artifacts"), exist_ok=True)
        output_filename = os.path.join(target_path, "artifacts", flnm)
        copy_files(
            source=artifact_path,
            source_filesystem=self._metadata_fs,
            destination=output_filename,
            destination_filesystem=LocalFileSystem(),
        )

    def _get_parameters(
        self, file_identifier: FileId, filesystem: Optional[FileSystem] = None
    ) -> Mapping[str, Parameter]:
        _ = self._get_metadata(file_identifier=file_identifier)
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

    def _get_metadata(self, file_identifier: FileId) -> Mapping[str, Metadata]:
        # Read from the metadata parquet document.
        if not self._check_file_id(file_identifier):
            raise TSFNFError(file_identifier=file_identifier)
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
        self, file_identifier: FileId, filesystem: Optional[FileSystem] = None
    ) -> Mapping[str, Metric]:
        _ = self._get_metadata(file_identifier=file_identifier)
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

    def _post_process_browse(self, browse_results: pd.DataFrame) -> pd.DataFrame:
        return browse_results

    def _check_file_id(self, file_identifier: FileId) -> bool:
        """Determine if a file id is in the metadata."""
        # Go get the root path for the file id.
        fl_info = self._metadata_fs.get_file_info(
            os.path.join(self._fs_output_location, file_identifier)
        )
        # 'Not' not found.
        return not fl_info.type == 0  # This is a File Not Found code.

    def _test_field_value(self, field: str, value: Metadata) -> bool:
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
