"""MLFlow Metadata Object.

This implements a Thing Store backed by an MLFlow object.
"""
import logging
import mlflow
import pandas as pd
import pyarrow.dataset as ds
import tempfile

from thethingstore.thing_store_base import (
    ThingStore,
)
from thethingstore.api.error import (
    ThingStoreFileNotFoundError as TSFNFError,
    ThingStoreNotAllowedError,
)
from thethingstore.types import Dataset, FileId, Parameter, Metadata, Metric
from mlflow import client
from pyarrow import Table
from pyarrow.fs import FileSystem, FileSelector
from typing import Any, Mapping, List, Optional, Union
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class MLFlowThingStore(ThingStore):
    """This leverages the MLFlow Python API.

    Note that MLFlow jsonifies parameters and thus will stringify values.
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        metadata_filesystem: Optional[FileSystem] = None,
        local_storage_folder: Optional[str] = None,
    ) -> None:
        super().__init__(
            metadata_filesystem=metadata_filesystem,
            local_storage_folder=local_storage_folder,
        )
        if tracking_uri is None:
            tracking_uri = f"file://{self._local_storage_folder}"
        self._mlflow_tracking_uri = tracking_uri
        self._mlflow_client = client.MlflowClient(
            tracking_uri=tracking_uri,
            registry_uri=None,  # Overload me to use a model registry.
        )

    def _load(
        self,
        file_identifier: FileId,
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
        if "data" not in self._list_artifacts(
            file_identifier=file_identifier, strip_data=False
        ):
            return None
        latest_run = self._get_metadata(
            file_identifier=file_identifier, post_process=False
        )
        # This might require some rework if a remote instance of mlflow is used.
        artifact_location = f"{latest_run['artifact_uri']}/data"
        return ds.dataset(artifact_location, filesystem=self._metadata_fs)

    def _log(
        self,
        dataset: Optional[Dataset] = None,
        parameters: Optional[Mapping[str, Parameter]] = None,
        metadata: Optional[Mapping[str, Optional[Metadata]]] = None,
        metrics: Optional[Mapping[str, Metric]] = None,
        artifacts_folder: Optional[str] = None,
    ) -> FileId:
        """Store a file in the Thing Store and associated information into the metadata.

        This will load your information into the Thing Store.

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
            If this is not unique this will raise an exception.
        """
        if metadata is None:
            metadata = {}
        if parameters is None:
            parameters = {}
        if metrics is None:
            metrics = {}
        _metadata = self._scrape_metadata(metadata)
        mlflow_experiment_name = _metadata["FILE_ID"]
        if not isinstance(mlflow_experiment_name, str):
            raise ThingStoreNotAllowedError(
                "Non-String-FILEID", "Nonstring FILEIDs are not implemented currently."
            )
        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name=mlflow_experiment_name)
        with tempfile.TemporaryDirectory() as t:
            with mlflow.start_run():
                mlflow.set_tags(_metadata)
                mlflow.log_params(parameters)
                mlflow.log_metrics(metrics)
                if dataset is not None:
                    ds.write_dataset(dataset, t, format="parquet")
                    mlflow.log_artifact(t, artifact_path="data")
                if artifacts_folder is not None:
                    mlflow.log_artifacts(
                        local_dir=artifacts_folder, artifact_path="artifacts"
                    )
        return mlflow_experiment_name

    def _list_artifacts(
        self, file_identifier: FileId, strip_data: bool = True
    ) -> List[str]:
        latest_run = self._get_metadata(
            file_identifier=file_identifier, post_process=False
        )
        artifacts = self._metadata_fs.get_file_info(
            FileSelector(urlparse(latest_run["artifact_uri"]).path, recursive=True)
        )
        if not artifacts:
            return []
        # This creates an initial set. Depending on how nested the
        #   dataset is multiple paths will need to be removed.
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
                if not _.path.endswith("artifacts")
            ]
        )
        _dataset_row = artifacts.query('base_name=="data"')
        # Do we need to account for data?
        if not _dataset_row.empty:  # We have an identified data source.
            # We are going to limit the dataset representation to one row.
            _dataset_prefix = _dataset_row.path.item()
            artifacts = pd.concat(
                [
                    artifacts.loc[~artifacts.path.str.startswith(_dataset_prefix)],
                    _dataset_row,
                ]
            )
        if strip_data:
            artifacts = artifacts.query('base_name!="data"')
        return artifacts.base_name.to_list()

    def _get_artifact(
        self, file_identifier: FileId, artifact_identifier: str, target_path: str
    ) -> None:
        """Copy an artifact locally.

        This copies an artifact from the Thing Store to a local path.

        Parameters
        ----------
        file_identifier: FileId
            This is a file identifier understood by the Thing Store.
        artifact_identifer: str
            This is the string key for the artifact (filepath in the folder).
        target_path: str
            This is where you wish to move the file.
        """
        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        mlflow.artifacts.download_artifacts(
            run_id=self._get_metadata(
                file_identifier=file_identifier, post_process=False
            )["run_id"],
            artifact_path=f"artifacts/{artifact_identifier}",
            dst_path=target_path,
        )

    def _get_parameters(self, file_identifier: FileId) -> Mapping[str, Parameter]:
        _ = self._get_metadata(file_identifier=file_identifier, post_process=False)
        params = {
            k.replace("params.", ""): v for k, v in _.items() if k.startswith("params.")
        }
        if not params:
            logger.warn(f"No parameters for FILE_ID@{file_identifier}")
            return {}
        return params

    def _get_metadata(
        self, file_identifier: FileId, post_process: bool = True
    ) -> Mapping[str, Metadata]:
        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        res = mlflow.search_experiments(filter_string=f'name="{file_identifier}"')
        if len(res) == 0:
            raise TSFNFError(file_identifier=file_identifier)
        run = mlflow.search_runs(experiment_ids=[res[0].experiment_id])
        # What is the latest run?
        latest_run = run.loc[run[["tags.FILE_VERSION"]].astype("float").idxmax(), :]
        if not post_process:
            return latest_run.to_dict(orient="records")[0]
        latest_run = self._post_process_browse(latest_run).to_dict(orient="records")[0]
        return latest_run

    def _get_metrics(self, file_identifier: FileId) -> Mapping[str, Metric]:
        _ = self._get_metadata(file_identifier=file_identifier, post_process=False)
        metrics = {
            k.replace("metrics.", ""): v
            for k, v in _.items()
            if k.startswith("metrics.")
        }
        if not metrics:
            logger.warn(f"No metrics for FILE_ID@{file_identifier}")
            return {}
        return metrics

    def _browse(self, **kwargs: Any) -> Table:
        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        experiments = mlflow.search_experiments(**kwargs)
        experiment_ids = [_.experiment_id for _ in experiments]
        return mlflow.search_runs(experiment_ids).reset_index(drop=True)

    def _check_file_id(self, file_identifier: FileId) -> bool:
        """Determine if a file id is in the metadata."""
        mlflow.set_tracking_uri(self._mlflow_tracking_uri)
        results = mlflow.search_experiments(filter_string=f'name="{file_identifier}"')
        return len(results) > 0

    def _post_process_browse(self, _browse_results: pd.DataFrame) -> pd.DataFrame:
        """Return post-processed metadata.

        This converts the MLFlow Metadata by converting all the tags
        to metadata elements.
        """
        # This list has a name (FILE_ID), tags (metadata), artifact location
        _column_list = _browse_results.columns
        column_list = [
            _
            for _ in _column_list
            if _.startswith("tags") and not _.startswith("tags.mlflow")
        ] + ["artifact_uri"]
        output = _browse_results[column_list].rename(
            columns=lambda x: x.replace("tags.", "")
        )
        output = output.drop(
            columns="artifact_uri",
        )
        if "FILE_VERSION" in output:
            output = output.astype({"FILE_VERSION": "int64"})
        return output

    def _test_field_value(self, field: str, value: Metadata) -> bool:
        filter_string = f'tags.{field}="{value}"'
        # raise Exception(filter_string)
        res = mlflow.search_runs(filter_string=filter_string)
        if len(res) == 0:
            return False
        else:
            return True
