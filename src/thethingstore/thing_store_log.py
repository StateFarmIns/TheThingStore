"""Hold the logging routine.

The logging routine is complex enough that it warrants being
represented by itself.
"""
import logging
import thethingstore.api.load as tsl
import os
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from copy import deepcopy
from datetime import datetime
from thethingstore.types import Dataset, Parameter, Metadata, Metric
from tempfile import TemporaryDirectory
from typing import Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from thethingstore.thing_store_base import ThingStore

logger = logging.getLogger(__name__)


def log(
    thing_store: "ThingStore",  # noqa: F821 - ThingStore isn't defined.
    dataset: Optional[Dataset] = None,
    parameters: Optional[Mapping[str, Parameter]] = None,
    metadata: Optional[Mapping[str, Optional[Metadata]]] = None,
    metrics: Optional[Mapping[str, Metric]] = None,
    artifacts_folder: Optional[str] = None,
    force: bool = False,
) -> str:
    _indicator_fields = {
        "TS_HAS_DATASET": dataset is not None,
        "TS_HAS_PARAMETERS": parameters is not None,
        "TS_HAS_METADATA": metadata is not None,
        "TS_HAS_METRICS": metrics is not None,
        "TS_HAS_ARTIFACTS": artifacts_folder is not None,
    }

    with TemporaryDirectory() as t:
        # Is this a FILE_ID?
        _dataset = _handle_dataset(
            dataset=dataset,
            thing_store=thing_store,
            temp_folder=t,
        )
        # Examine the metadata and handle edge cases
        metadata = _handle_metadata(
            metadata=metadata,
            thing_store=thing_store,
            force=force,
        )
        # This will be removed later and that's fine.
        # This helps mypy understand this is ok.
        assert isinstance(metadata, dict)  # nosec
        metadata.update(_indicator_fields)
        return thing_store._log(
            dataset=_dataset,
            parameters=parameters,
            metadata=metadata,
            metrics=metrics,
            artifacts_folder=artifacts_folder,
        )


def _handle_dataset(
    dataset: Dataset,
    thing_store: "ThingStore",  # noqa: F821 - ThingStore isn't defined.
    temp_folder: str,
) -> Dataset:
    """Handle a dataset.

    This is a helper function that just consolidates the logic
    behind handling the different types of datasets that the
    Thing Store may represent."""
    if dataset is not None:  # There *is* a dataset.
        _, loadtype = tsl._get_info(dataset_or_filepaths=dataset)
        if loadtype == "fileid":
            # Does this FILE_ID exist in the thing store?
            if thing_store._check_file_id(file_identifier=dataset):
                # Good, let's not copy it!
                _dataset = dataset
        # TODO: validate the change from if to elif does not break anything.
        elif isinstance(dataset, str):
            if loadtype == "shape":
                try:
                    import geopandas as gp
                except ImportError:
                    raise Exception("Please install geopandas to load shapes.")

                _dataset_gp = gp.read_file(dataset)
                _, flname = os.path.split(dataset)
                _dataset_gp.to_parquet(os.path.join(temp_folder, flname))
                _dataset = ds.dataset(os.path.join(temp_folder, flname))
            else:
                _dataset = ds.dataset(dataset)
        elif isinstance(dataset, pd.DataFrame):
            _dataset = pa.Table.from_pandas(
                dataset, schema=pa.Schema.from_pandas(dataset)
            )
        else:
            _dataset = dataset
        if not isinstance(_dataset, (pa.Table, ds.Dataset)):
            raise TypeError("Only pyarrow compatible.")
    else:
        _dataset = None
    return _dataset


def _handle_metadata(
    metadata: Optional[Mapping[str, Optional[Metadata]]],
    thing_store: "ThingStore",
    force: bool = False,
) -> Mapping[str, Optional[Metadata]]:
    """Handle metadata.

    This is a helper function that just consolidates the logic
    behind handling some of the different edge cases potentially
    seen in metadata.

    FILE_ID Treatment: FILE identifiers and FILE_VERSIONS pairs
        uniquely identify files.
    """
    # First, do I *have* metadata.
    if not isinstance(metadata, dict):  # No?
        return {}  # Well... I do now.
    metadata = deepcopy(metadata)  # Be sure not to stomp on customer metadata.
    # Handle the FILE_ID appropriately.
    fl_id: str = str(metadata.get("FILE_ID"))
    # Handle the FILE_VERSION appropriately
    fl_ver: int = int(float(metadata.get("FILE_VERSION", 1)))
    if fl_id != "None":  # I was given a FILE_ID.
        if thing_store._check_file_id(fl_id):  # It exists in the thing store.
            # Get the most current metadata.
            _remote_metadata = thing_store.get_metadata(fl_id)
            # Note this gets the file version which defaults to zero.
            managed_fl_ver: int = int(float(_remote_metadata.get("FILE_VERSION", 0)))
            if fl_ver <= 0:  # If the *requested* version is 0 or lower... explode.
                raise NotImplementedError(
                    "File versions of 0 and below are not officially supported."
                )
            elif (
                fl_ver <= managed_fl_ver
            ) and not force:  # Need to bump to avoid the stomp.
                logger.warn(
                    f"FILE_VERSION ALREADY EXISTS: AMENDED TO {managed_fl_ver + 1}"
                )
                metadata["FILE_VERSION"] = managed_fl_ver + 1  # type: ignore
            elif force:  # Someday this will be ok. Not today ISIS.
                raise NotImplementedError(
                    "This requires the capacity to modify the metadata safely"
                )
    dataset_date = metadata.get("DATASET_DATE")
    if dataset_date is None:
        metadata["DATASET_DATE"] = datetime.now()
    return metadata
