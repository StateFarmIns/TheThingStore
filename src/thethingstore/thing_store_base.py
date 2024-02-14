"""Base Thing Store Object

Regardless of implementation the basic thoughts behind the metadata
dataset include:
* The capability to *add* a file to the dataset.
* The capability to *remove* from the dataset by file identifier.
* The capability to *load* from the dataset by file identifier or schema.
"""
import logging
import numpy as np
import os
import pandas as pd
import pyarrow.dataset as ds
import uuid
from thethingstore.api import load as tsl, error as tse, data_hash as tsh
from thethingstore.types import Dataset, FileId, Parameter, Metadata, Metric
from thethingstore.thing_store_elements import Metadata as MetadataElements
from thethingstore.thing_store_log import log as tslog
from pyarrow import Table
from pyarrow.fs import FileSystem, LocalFileSystem
from tempfile import TemporaryDirectory
from typing import Any, Mapping, List, Optional, Type, Union


logger = logging.getLogger(__name__)


# TODO: What base would be appropriate? ABC?
class ThingStore:
    """Implements API specification.

    A ThingStore should be able to load, log, browse, list / get
    artifacts, and *copy* from one ThingStore to another.
    """

    def __init__(
        self,
        metadata_filesystem: Optional[FileSystem] = None,
        local_storage_folder: Optional[str] = None,
    ) -> None:
        self._tempdir = None
        if local_storage_folder is None:
            self._tempdir = TemporaryDirectory()
            local_storage_folder = self._tempdir.name
        self._local_storage_folder = local_storage_folder
        self._local_fs = LocalFileSystem()
        if metadata_filesystem is None:
            metadata_filesystem = self._local_fs
        self._metadata_fs = metadata_filesystem

    def _load(
        self,
        file_identifier: FileId,
    ) -> Union[Dataset, None]:
        """Return dataset paths identified by FILE_IDs.

        This queries the metadata to get appropriate file
        handles for datasets managed by the Thing Store.

        Parameters
        ----------
        file_identifier: FileId
            This is a file identifier understood by the Thing Store.
        **kwargs
            These are forwarded into _load from load.

        Returns
        -------
        filepaths: List[str]
        """
        raise NotImplementedError("Overload me.")

    def load(
        self,
        file_identifier: Union[FileId, List[FileId], Dataset, List[Dataset]],
        output_format: str = "pandas",
        **kwargs: Any,
    ) -> Dataset:
        """Read a file from the Thing Store.

        This returns either a representation of the dataset or the
        dataset itself.

        Parameters
        ----------
        file_identifier: FileId
            This is a file identifier understood by the the Thing Store.
        output_format: str = 'pandas
            Specify how to return the data.

        Returns
        -------
        output_data: Dataset
            Output data in the form requested.
        """
        # 1. Identify the filetypes to load.
        file_handles, filetype = tsl._get_info(file_identifier)
        if filetype == "fileid":  # This is a file id.
            # FILE_ID live in the thing store. Go look for it.
            file_handles = [self._load(file_identifier=_) for _ in file_handles]
            # This is now hard coded.
            filetype = "dataset"
        load_dataset_kwargs = kwargs.get("load_dataset_kwargs", {})
        if load_dataset_kwargs is None:
            load_dataset_kwargs = {}
        if "filesystem" not in load_dataset_kwargs:
            load_dataset_kwargs["filesystem"] = self._metadata_fs
        load_table_kwargs = kwargs.get("load_table_kwargs", {})
        if load_table_kwargs is None:
            load_table_kwargs = {}
        load_pandas_kwargs = kwargs.get("load_pandas_kwargs", {})
        if load_pandas_kwargs is None:
            load_pandas_kwargs = {}
        return tsl.load(
            dataset_or_filepaths=file_handles,
            dataset_type=filetype,
            output_format=output_format,
            load_dataset_kwargs=load_dataset_kwargs,
            load_table_kwargs=load_table_kwargs,
            load_pandas_kwargs=load_pandas_kwargs,
        )

    def _log(
        self,
        dataset: Optional[Dataset] = None,
        parameters: Optional[Mapping[str, Parameter]] = None,
        metadata: Optional[Mapping[str, Optional[Metadata]]] = None,
        metrics: Optional[Mapping[str, Metric]] = None,
        artifacts_folder: Optional[str] = None,
    ) -> FileId:
        raise NotImplementedError("Overload me!")

    def log(
        self,
        dataset: Optional[Dataset] = None,
        parameters: Optional[Mapping[str, Parameter]] = None,
        metadata: Optional[Mapping[str, Optional[Metadata]]] = None,
        metrics: Optional[Mapping[str, Metric]] = None,
        artifacts_folder: Optional[str] = None,
        force: bool = False,
    ) -> FileId:
        """Store a file in the Thing Store and associated metadata.

        This will load your information into the Thing Store.
        (i.e. the dataset is an existing FILE_ID) this will reuse
        that file.

        Parameters
        ----------
        dataset: Optional[Dataset] = None
            This is understandable as a dataset.
        parameters: Optional[Mapping[str, Parameter]] = None
            These are parameters that may be logged with this file.
        metadata: Optional[Mapping[str, Metadata]] = None
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
        """
        return tslog(
            thing_store=self,
            dataset=dataset,
            parameters=parameters,
            metadata=metadata,
            metrics=metrics,
            artifacts_folder=artifacts_folder,
            force=force,
        )

    def _browse(self, **kwargs: Any) -> Table:
        """Return a raw representation of the metadata dataset."""
        raise NotImplementedError("Overload me.")

    def browse(self, **kwargs: Any) -> pd.DataFrame:
        """Return a processed set of metadata.

        This returns a processed metadata dataset.
        """
        return self._post_process_browse(self._browse(**kwargs)).reset_index(drop=True)

    def _list_artifacts(self, file_identifier: FileId) -> List[str]:
        """Return artifact identifiers associated with a file.

        This determines all artifacts associated with a file and
        returns those as a list of strings.

        Parameters
        ----------
        file_identifier: FileId
            This is a file identifier understood by the Thing Store.

        Returns
        -------
        artifact_identifiers: List[str]
            The keys for the artifacts associated with this file.
        """
        raise NotImplementedError("Overload me.")

    def list_artifacts(
        self,
        file_identifier: FileId,
    ) -> List[str]:
        """Return artifact identifiers associated with a file.

        This determines all artifacts associated with a file and
        returns those as a list of strings.

        Parameters
        ----------
        file_identifier: FileId
            This is a file identifier understood by the Thing Store.

        Returns
        -------
        artifact_identifiers: List[str]
            The keys for the artifacts associated with this file.
            None if no artifacts exist.
        """
        return self._list_artifacts(file_identifier)

    def _check_file_id(self, file_identifier: FileId) -> bool:
        """Determine if a file id is in the thing store."""
        raise NotImplementedError

    def _unique_file_id(self) -> str:
        """Return a unique file id in the thing store."""
        cnt = 0
        # TODO: Betterify.
        # For loop with an exception instead with internal return?
        while True:
            cnt += 1
            newflid = uuid.uuid4().hex
            if not self._check_file_id(newflid):
                return newflid
            if cnt > 10:
                raise Exception("SOMETHING IS WRONG!")

    def _scrape_metadata(
        self, metadata_elements: Mapping[str, Optional[Metadata]]
    ) -> Mapping[str, Optional[Metadata]]:
        """Investigate, update, and return standard metadata."""
        # Update the metadata with default elements.
        # Note the nosec? Bandit is cranky about these, Mypy is cranky
        #   without them... Come on guys, get it together.
        assert isinstance(metadata_elements, dict)  # nosec
        _metadata = MetadataElements(**metadata_elements).dict()
        assert isinstance(_metadata, dict)  # nosec
        ###########
        # FILE_ID #
        ###########
        # If you pass one and it is none, this puts in a unique file id.
        if _metadata["FILE_ID"] is None:  # Not user supplied
            # Get a unique FILE_ID
            _metadata["FILE_ID"] = self._unique_file_id()
        ################
        # FILE_VERSION #
        ################
        if _metadata["FILE_VERSION"] is None:  # Not user supplied
            # TODO: Is this ever going to get hit anymore?
            # Whatever the case, this should be moved to metadata_elements
            # Get a FILE_VERSION
            _metadata["FILE_VERSION"] = 1
        return _metadata

    def _post_process_browse(self, browse_results: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def _get_artifact(
        self, file_identifier: FileId, artifact_identifier: str, target_path: str
    ) -> None:
        """Copy an artifact locally.

        This copies an artifact from the thing store to a local path.

        Parameters
        ----------
        file_identifier: FileId
            This is a file identifier understood by the Thing Store.
        artifact_identifer: str
            This is the string key for the artifact (filepath in the folder).
        target_path: str
            This is where you wish to move the file.
        """
        raise NotImplementedError("Overload me!")

    def get_artifact(
        self, file_identifier: FileId, artifact_identifier: str, target_path: str
    ) -> None:
        """Copy an artifact locally.

        This copies an artifact from the thing store to a local path.

        Parameters
        ----------
        file_identifier: FileId
            This is a file identifier understood by the Thing Store.
        artifact_identifer: str
            This is the string key for the artifact (filepath in the folder).
        target_path: str
            This is where you wish to move the file.
        """
        self._get_artifact(
            file_identifier=file_identifier,
            artifact_identifier=artifact_identifier,
            target_path=target_path,
        )

    def get_artifacts(self, file_identifier: FileId, target_path: str) -> None:
        """Copy artifacts locally.

        This copies all artifacts (for a specific FILE_ID) from the
        Thing Store to a local path.

        This will make an 'artifacts' folder at the location you specify,
        to whit <target_path>/artifacts/<downloaded contents>.

        If no artifacts exist, no changes will be made.

        Parameters
        ----------
        file_identifier: FileId
            This is a file identifier understood by the Thing Store.
        target_path: str
            This is where you wish to move the artifacts.
        """
        artifacts = self.list_artifacts(file_identifier=file_identifier)
        if not artifacts:
            logger.warning("No artifacts exist. Nothing was done.")
            return None
        # This better be a directory!
        os.makedirs(target_path, exist_ok=True)
        if not os.path.isdir(target_path):
            raise RuntimeError(f"Cannot overwrite existing file @{target_path}")
        for fl in self.list_artifacts(file_identifier=file_identifier):
            self.get_artifact(
                file_identifier=file_identifier,
                artifact_identifier=fl,
                target_path=target_path,
            )

    def _get_parameters(self, file_identifier: FileId) -> Mapping[str, Parameter]:
        raise NotImplementedError

    def get_parameters(self, file_identifier: FileId) -> Mapping[str, Parameter]:
        """Return parameters used to produce a FILE_ID.

        This drops any parameters which are NaN or Null.

        Parameters
        ----------
        file_identifier: FileId
            A file identifier understood by the Thing Store.

        Returns
        -------
        parameters: Mapping[str, Parameter]
            The set of parameters used to produce the FILE_ID.
            None if no parameters were used.
        """
        params = self._get_parameters(file_identifier=file_identifier)
        params = {k: v for k, v in params.items() if v is not None}
        return params  # type: ignore  # Figure out why mypy is getting tetchy.

    def _get_metadata(self, file_identifier: FileId) -> Mapping[str, Metadata]:
        raise NotImplementedError("Overload me!")

    def get_metadata(
        self, file_identifier: FileId, drop_none: bool = True
    ) -> Mapping[str, Metadata]:
        """Return latest metadata associated with a FILE_ID.

        Note that this, by default, does not return any fields with None.

        Parameters
        ----------
        file_identifier: FileId
            A file identifier understood by the Thing Store.
        drop_none: bool = True
            Whether to remove fields with None. For consistency
            across implementations this defaults to True.
            This means that columns in the metadata which are
            missing elements are not represented as None.

        Returns
        -------
        metadata: Mapping[str, Metadata]
            The most up to date metadata for the FILE_ID.
            None if no metadata exists.
        """
        _metadata = self._get_metadata(file_identifier=file_identifier)
        if drop_none:
            _metadata = {k: v for k, v in _metadata.items() if v is not None}
        # MyPy doesn't like dict unpacking. Oh well.
        return MetadataElements(**_metadata).dict()  # type: ignore

    def _get_metrics(self, file_identifier: FileId) -> Mapping[str, Metric]:
        raise NotImplementedError("Overload me!")

    def get_metrics(self, file_identifier: FileId) -> Mapping[str, Metric]:
        """Return metrics associated with a FILE_ID.

        This drops any metrics which return NaN or Null.

        Parameters
        ----------
        file_identifier: FileId
            A file identifier understood by the Thing Store.

        Returns
        -------
        metrics: Union[Mapping[str, Metric], Type[None]]
            The set of metrics associated with the FILE_ID.
            None if no metrics exist.
        """
        metrics = self._get_metrics(file_identifier=file_identifier)
        metrics = {
            k: v for k, v in metrics.items() if (v is not None and not np.isnan(v))
        }
        return metrics

    def get_dataset(
        self, file_identifier: FileId, **kwargs: Any
    ) -> Union[ds.Dataset, Type[None]]:
        """Return file identifier as a PyArrow dataset.

        This leverages class methods to pull back the location of
        the dataset; this will default the arguments to appropriate
        values for the filesystem by substituting the filesystem
        for the metadata if it's not specified. That may be
        explicitly overridden by passing `filesystem=None`.

        Parameters
        ----------
        file_identifier: FileId
            A file identifier understood by the Thing Store.
        **kwargs
            Keyword arguments forwarded into the constructor
            for a PyArrow Dataset.

        Returns
        -------
        dataset: Union[ds.Dataset, Type[None]]
            The dataset associated with the FILE_ID if one exists.
            None if no dataset is associated.
        """
        file_handles, filetype = tsl._get_info(file_identifier)
        if not filetype == "fileid":  # This is not a file id.
            raise KeyError("Can only request a dataset by FILE_ID")
        # FILE_ID live in the Thing Store. Go look for it.
        file_handles = [self._load(file_identifier=_) for _ in file_handles]
        file_handles = [_ for _ in file_handles if (_ is not None and _)]
        # Validate this behavior works.
        if not file_handles:
            return None
        return ds.dataset(file_handles, **kwargs)

    def get_metadata_hash(self, return_type: str = "hex") -> Union[str, bytes, int]:
        """Returns a md5 digest of the latest metadata dataset

        Parameters
        ----------
        return_type: str
            This is the return type for the hash.
            Currently implemented types:
            - hex
            - bytes
            - int | integer

        Returns
        ----------
        hash: Union[str, bytes, int]
            Metadata dataset md5 hash
        """
        metadata = self.browse()
        return tsh.dataset_digest(df=metadata, return_type=return_type)

    def copy(  # noqa:C901 This is mildly complex, but it's not hard to follow.
        self,
        file_identifier: FileId,
        thing_store: "ThingStore",
        force: bool = False,
    ) -> None:
        """Copy a FileId **from another** Thing Store.

        This does nothing if the file is present and up to date.

        TODO: Break this to copy from / copy to.

        Parameters
        ----------
        file_identifier: FileId
            A file identifier understood by the Thing Store.
        thing_store: ThingStore
            The Thing Store which contains the file.
        force: bool = False
            Just do it. Swoosh.
        """
        # 1. Does the file id exist in the remote? What's the latest version?
        remote_version = thing_store._get_metadata(file_identifier=file_identifier)
        # 2. Does it exist locally?
        try:
            local_version = self._get_metadata(file_identifier=file_identifier)
        except tse.ThingStoreFileNotFoundError:
            local_version = {}
        local_keys = set(local_version.keys())
        remote_keys = set(remote_version.keys())
        # 3. Do they share the same metadata?
        if not local_version:  # No local version exists.
            up_to_date = False
        elif "FILE_VERSION" in local_keys.intersection(
            remote_keys
        ):  # Can check version.
            if local_version["FILE_VERSION"] < remote_version["FILE_VERSION"]:
                up_to_date = False
            elif local_version["FILE_VERSION"] == remote_version["FILE_VERSION"]:
                up_to_date = True
            elif local_version["FILE_VERSION"] > remote_version["FILE_VERSION"]:
                logger.warning(
                    f"{file_identifier} higher local file version than remote."
                )
                up_to_date = True
        else:
            warn_message = """File Version cannot be determind.

            I was unable to determine whether or not the file version
            in the remote data was more recent than local data. If you
            would like to force a copy from the Thing Store please specify
            `force=True` in the copy command.

            Local Thing Store
            --------------\n{local_version}

            Remote Thing Store
            ---------------\n{remote_version}
            """
            if force:
                up_to_date = False
            else:
                up_to_date = True
                logger.warning(warn_message)
        if up_to_date:
            _msg_head = f"FILE ID: {file_identifier}\n\t"
            logger.warn(
                f"{_msg_head}No Copy Performed: The latest version appears up to date."
            )
            return
        # 4. Get all the information for this FILE_ID.
        #########
        # Data. #
        #########
        # The dataset is identified in the remote thing store.
        _dataset = thing_store.get_dataset(file_identifier=file_identifier)
        ###############
        # Parameters. #
        ###############
        _params = thing_store.get_parameters(file_identifier=file_identifier)
        #############
        # Metadata. #
        #############
        _metadata = thing_store.get_metadata(file_identifier=file_identifier)
        ############
        # Metrics. #
        ############
        _metrics = thing_store.get_metrics(file_identifier=file_identifier)
        ##############
        # Artifacts. #
        ##############
        _artifacts = thing_store.list_artifacts(file_identifier=file_identifier)
        if _artifacts is None:
            _artifacts = []
        try:
            if _artifacts:
                with TemporaryDirectory() as t:
                    thing_store.get_artifacts(
                        file_identifier=file_identifier, target_path=t
                    )
                    self.log(
                        dataset=_dataset,
                        parameters=_params,
                        metadata=_metadata,
                        metrics=_metrics,
                        artifacts_folder=os.path.join(t, "artifacts"),
                    )
            else:
                self.log(
                    dataset=_dataset,
                    parameters=_params,
                    metadata=_metadata,
                    metrics=_metrics,
                )
        except BaseException as e:
            raise tse.ThingStoreGeneralError(
                f"""Unable to copy to remote.",
                Type dataset: {type(_dataset)}
                Parameters: {_params}
                Metadata: {_metadata}
                Metrics: {_metrics}
                Artifacts: {_artifacts}
                """,
            ) from e
