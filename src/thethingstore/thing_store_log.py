"""Hold the logging routine.

The logging routine is complex enough that it warrants being
represented by itself.
"""
import importlib.util
import inspect
import logging
import tempfile
import thethingstore.api.load as tsl
import thethingstore.api.save as tss
import os
import pyarrow as pa
import pyarrow.dataset as ds
import shutil
from copy import deepcopy
from datetime import datetime
from thethingstore.types import Dataset, FileId, Parameter, Metadata, Metric, Thing
from tempfile import TemporaryDirectory
from typing import Mapping, Optional, Union, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from thethingstore.thing_store_base import ThingStore

logger = logging.getLogger(__name__)


def log(
    thing_store: "ThingStore",  # noqa: F821 - ThingStore isn't defined.
    dataset: Optional[Dataset] = None,
    parameters: Optional[Mapping[str, Parameter]] = None,
    metadata: Optional[Mapping[str, Optional[Metadata]]] = None,  # type: ignore
    metrics: Optional[Mapping[str, Metric]] = None,  # type: ignore
    artifacts_folder: Optional[str] = None,
    embedding: Optional[Dataset] = None,
    force: bool = False,
    **kwargs: dict,
) -> str:
    _indicator_fields = {
        "TS_HAS_DATASET": dataset is not None,
        "TS_HAS_PARAMETERS": parameters is not None,
        "TS_HAS_METADATA": metadata is not None,
        "TS_HAS_METRICS": metrics is not None,
        "TS_HAS_ARTIFACTS": artifacts_folder is not None,
        "TS_HAS_EMBEDDING": embedding is not None,
    }

    with TemporaryDirectory() as t:
        # Is this a FILE_ID?
        # Tim's thoughts: This pattern can have the 'get by fileid'
        #   abstracted to enable thingpointer.
        # Then the pattern is Thing::by_component() yielding a loop of _handle_{component}(x) over the
        #   list of components.
        # In _handle_{component} the pattern is 'get_by_fileid if fileid' and
        #   then component specific logic.
        _dataset = _handle_dataset(
            dataset=dataset,
            thing_store=thing_store,
            temp_folder=t,
        )
        # Examine the metadata and handle edge cases
        _metadata = _handle_metadata(
            metadata=metadata,
            thing_store=thing_store,
            force=force,
        )
        # This will be removed later and that's fine.
        # This helps mypy understand this is ok.
        if _metadata is None:
            _metadata = {}
        assert isinstance(_metadata, dict)  # nosec
        _metadata.update(_indicator_fields)
        # This is UNOFFICIAL support for this.
        # Check for artifacts here.
        if "artifacts" in kwargs and artifacts_folder is None:
            os.makedirs(os.path.join(t, "artifacts"))
            tss.save(kwargs["artifacts"], os.path.join(t, "artifacts", "artifact"))
            _artifacts_folder: Optional[str] = os.path.join(t, "artifacts")
        else:
            _artifacts_folder = artifacts_folder  # type: ignore
        _embedding = _handle_dataset(  # type: ignore
            dataset=embedding,
            thing_store=thing_store,  # type: ignore
            temp_folder=t,  # type: ignore
        )
        return thing_store._log(
            dataset=_dataset,
            parameters=parameters,
            metadata=_metadata,
            metrics=metrics,
            artifacts_folder=_artifacts_folder,
            embedding=_embedding,
        )


def _handle_dataset(  # noqa: C901 - flake is a 'clean code' Andy
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
        # TODO: For Ibis implementation here this will replace PDDataFrame.
        if loadtype == "fileid":
            # Does this FILE_ID exist in the thing store?
            if thing_store._check_file_id(file_identifier=dataset):
                # Good, let's not copy it!
                _dataset = dataset
        # TODO: validate the change from if to elif does not break anything.
        else:
            # These conditional imports allow for testing without blowing
            #   up if things like torch are not installed
            try:
                # flake doesn't like conditional imports
                import pandas as pd  # noqa: F401

                PDDataFrame = pd.DataFrame
            except ImportError:
                PDDataFrame = None
            try:
                import torch

                TorchTensor = torch.Tensor
            except ImportError:
                TorchTensor = None
            if isinstance(dataset, str):
                # This one is special
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
            elif PDDataFrame is not None and isinstance(dataset, PDDataFrame):
                _dataset = pa.Table.from_pandas(
                    dataset, schema=pa.Schema.from_pandas(dataset)
                )
            elif TorchTensor is not None and isinstance(dataset, TorchTensor):
                try:
                    import ibis
                except ImportError:
                    raise Exception("Please install ibis to work with embeddings.")
                _dataset = ibis.memtable(dataset).to_pyarrow()
            else:
                _dataset = dataset
        if not isinstance(_dataset, (pa.Table, ds.Dataset)):
            raise TypeError("Only pyarrow compatible.")
    else:
        _dataset = None
    return _dataset


def _handle_metadata(
    metadata: Optional[Mapping[str, Optional[Metadata]]],  # type: ignore
    thing_store: "ThingStore",
    force: bool = False,
) -> Mapping[str, Optional[Metadata]]:  # type: ignore
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
    # Handle the DATASET_DATE appropriately
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


def _get(value: Parameter) -> Any:
    # This helper function is reused below in the import phase.
    if value.default == inspect._empty:
        return None
    else:
        return value.default


def log_function(  # noqa: C901
    thing_store: "ThingStore", python_file_path: str, dry_fire: bool = True
) -> Union[FileId, str]:
    """Log a functional Thing.

    This is a convenience function that makes it easy to save a Python
    function into the ThingStore.

    This function will validate that a Python file has something
    which can be published as a functional Thing.

    It performs the following checks:
        * Does the Python file exist?
        * Does the file have a workflow function?
        * Are there components (specified as workflow_{X}) in the set:
            * Metadata
            * Metrics
            * Embedding
            * Dataset
        * Does this output a Thing? (Not required, strongly encouraged.)

    It stores the function in a temporary data layer.

    **IF** `dry_fire==True` it will then publish to the provided thing store.

    Parameters
    ----------
    thing_store: 'ThingStore'
        This is a ThingStore compliant data layer.
    python_file_path: str
        This is a filepath of a file ending in `.py`.
    dry_fire: bool = True
        Whether to push to the layer.

    Returns
    -------
    logged_function_fileid: Optional[FileId]
        If `dry_fire==False` this will save to the thing store and
        return a fileid representing the function.
    """
    # 1. Does the file exist?
    if not os.path.exists(python_file_path):
        raise FileNotFoundError(
            f"""Functional Thing Exception
        You attempted to log a workflow that does not appear to exist.

        Please investigate the loaded path below.

        Loaded Path
        -----------\n{python_file_path}

        Current Working Directory
        -------------------------\n{os.getcwd()}
        """
        )
    if not os.path.isfile(python_file_path):
        raise TypeError(
            f"""Functional Thing Exception
        You attempted to log a workflow that is not a file.

        Please investigate the loaded path below.

        Loaded Path
        -----------\n{python_file_path}
        """
        )
    # 2. Does the file contain a workflow function?
    try:
        # This uses standard Python import libraries
        #   to bring down a Python file and read it into
        #   scope. After this it's available as a module.
        _, py_file_name = os.path.split(python_file_path)
        spec = importlib.util.spec_from_file_location(
            py_file_name.replace(".py", ""), python_file_path.replace(".py", "") + ".py"
        )
        assert spec is not None  # nosec
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)  # type: ignore
    except BaseException as e:
        raise ImportError(
            f"""Functional Thing Exception
        I could not import the workflow you wish to log.

        Please attempt to load the module externally to troubleshoot.

        Loaded Path
        -----------\n{python_file_path}
        """
        ) from e
    if not hasattr(_module, "workflow"):
        raise KeyError(
            f"""Functional Thing Exception
        Your workflow file does not appear to have a workflow function.

        Please check and ensure your function is named `def workflow`

        Loaded Path
        -----------\n{python_file_path}
        """
        )
    # 3. Does the file contain Thing components?
    #   Note we skip parameters, artifacts, and function!
    component_dict: dict[str, Any] = {
        _: getattr(_module, f"workflow_{_}", None)
        for _ in (
            "dataset",
            "metadata",
            "metrics",
            "embedding",
        )
    }
    # The parameters we extract here.
    _params = inspect.signature(_module.workflow).parameters
    component_dict["parameters"] = {k: _get(v) for k, v in _params.items()}
    # Here we *could* allow artifacts and just add the function to them.
    # We are not going to, because Tim is lazy and doesn't want to implement that!
    # 4. Does this return a Thing?
    returns_a_thing = inspect.signature(_module.workflow).return_annotation == Thing
    if not returns_a_thing:
        logger.warning("Returning a Thing is encouraged.")
    # 5. Can the file be logged locally?
    with tempfile.TemporaryDirectory() as t:
        os.makedirs(f"{t}/artifacts/function")
        os.makedirs(f"{t}/thingstore")
        from thethingstore import FileSystemThingStore
        from pyarrow.fs import LocalFileSystem

        data_layer = FileSystemThingStore(
            metadata_filesystem=LocalFileSystem(), managed_location=f"{t}/thingstore"
        )
        shutil.copy(python_file_path, os.path.join(t, "artifacts/function/workflow.py"))
        component_dict.update(artifacts_folder=os.path.join(t, "artifacts"))
        if component_dict["metadata"] is None:
            component_dict["metadata"] = {}
        # Come on mypy get it together
        assert component_dict["metadata"] is not None  # nosec
        component_dict["metadata"].update(TS_HAS_FUNCTION=True)
        flid = data_layer.log(**component_dict)
        # 5. Now, validate it.
        _ = data_layer.get_function(flid)
        # TODO: More validation here.
        # Ok, send it!
        try:
            if not dry_fire:
                return thing_store.log(**component_dict)
            else:
                return f"Dry Run Success! fileid: {flid} published {_} to a temporary data layer."
        except BaseException:
            raise Exception("what", data_layer.get_metadata(flid), flid, component_dict)
