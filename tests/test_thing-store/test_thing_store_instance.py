"""Test concrete implementations of Thing Store."""
import itertools
import os
import pandas as pd
import pyarrow as pa
import pytest
import uuid
import pyarrow.dataset as ds
from datetime import datetime
from pyarrow.fs import LocalFileSystem, S3FileSystem
from thethingstore.thing_store_base import ThingStore
from thethingstore.thing_store_elements import Metadata
from thethingstore.thing_store_mlflow import MLFlowThingStore
from thethingstore.thing_store_pa_fs import FileSystemThingStore


thing_store_sets = [
    (
        MLFlowThingStore,
        {
            "tracking_uri": "tracking",
            "local_storage_folder": "local_storage",
        },
        "MLFlow",
    ),
    (
        FileSystemThingStore,
        {
            "metadata_filesystem": LocalFileSystem(),
            "metadata_file": "metadata.parquet",
            "metadata_lockfile": "metadata-lockfile.parquet",
            "output_location": "output",
        },
        "LocalFilesystem",
    ),
    (
        FileSystemThingStore,
        {
            "metadata_filesystem": S3FileSystem(
                endpoint_override="http://localhost:5000", allow_bucket_creation=True
            ),
            "metadata_file": "metadata.parquet",
            "metadata_lockfile": "metadata-lockfile.parquet",
            "output_location": "output",
        },
        "S3Filesystem",
    ),
]


def _treat_params(params, folder):
    params = params.copy()
    newfolder = f"{folder}/{uuid.uuid4().hex}"
    if (
        "metadata_filesystem" in params
        and isinstance(params["metadata_filesystem"], S3FileSystem)
        and newfolder[1:]
        and params["metadata_filesystem"].get_file_info(newfolder[1:]).type.name
        == "NotFound"
    ):
        params["metadata_filesystem"].create_dir(newfolder[1:], recursive=True)
    for param in params:
        if not param == "metadata_filesystem":
            params[param] = f"{newfolder}/{params[param]}"
    return params


@pytest.fixture(
    params=[_ for _ in thing_store_sets if _[2] != "S3Filesystem"],
    ids=lambda x: f"Local:{x[2]}",
)
@pytest.mark.usefixtures("test_temporary_folder")
def local_thing_store(request, test_temporary_folder):
    # This creates a local thing_store.
    thing_store, params, test_name = request.param
    new_params = _treat_params(params, test_temporary_folder)
    yield thing_store(**new_params)


@pytest.fixture(params=thing_store_sets, ids=lambda x: f"Remote:{x[2]}")
@pytest.mark.usefixtures("test_temporary_folder")
def remote_thing_store(request, test_temporary_folder):
    # This creates a local thing_store.
    thing_store, params, test_name = request.param
    new_params = _treat_params(params, test_temporary_folder)
    yield thing_store(**new_params)


@pytest.mark.usefixtures("integration_test", "remote_thing_store", "local_thing_store")
def test_thing_store(integration_test, remote_thing_store, local_thing_store):
    """Run the integration test at tests/conftest.py"""
    integration_test(remote_thing_store, local_thing_store)


####################################################################
#                          Test Logging                            #
# ---------------------------------------------------------------- #
# This section tests the capability of the thing store to          #
#   log and recall components.                                     #
# This uses the full set of different instances of thing           #
#   store and the full set of combinations of different            #
#   components.                                                    #
####################################################################

components = {
    "dataset": pd.DataFrame([{"silly": "example"}]),
    "parameters": {"i": "have", "simple": "parameters"},
    "metrics": {"simple": 0, "metrics": 1.2},
    "artifacts_folder": None,  # This is replaced
    "metadata": {"simple": "metadata", "for": "testing"},
}

_component_combinations = []
for _ in range(1, len(components)):
    _component_combinations += list(itertools.combinations(components, _))


@pytest.fixture(
    params=_component_combinations,
    ids=lambda x: "-".join(x),
)
def component_combination(request):
    "This is a fixture which creates combinations of components."
    return request.param


def _loader(
    thing_store: ThingStore, file_id: str, component: str, base_path: str
) -> None:
    """Intelligently attempt to load and test a 'thing'"""
    if component == "dataset":
        dataset = thing_store.get_dataset(file_identifier=file_id)
        assert isinstance(dataset, ds.Dataset)
        assert dataset.schema == pa.schema({"silly": "string"})
        assert dataset.to_table().to_pylist() == [{"silly": "example"}]
    elif component == "parameters":
        parameters = thing_store.get_parameters(file_identifier=file_id)
        assert isinstance(parameters, dict)
        assert parameters == components["parameters"]
    elif component == "metrics":
        metrics = thing_store.get_metrics(file_identifier=file_id)
        assert isinstance(metrics, dict)
        assert metrics == components["metrics"]
    elif component == "artifacts_folder":
        artifacts = thing_store.get_artifacts(
            file_identifier=file_id, target_path=base_path
        )
        assert artifacts is None
        assert set(os.listdir(f"{base_path}/artifacts")) == {"numpy.npy", "list.txt"}
        # Some day consider testing artifact loading here?
    elif component == "metadata":
        _metadata = thing_store.get_metadata(file_identifier=file_id)
        assert isinstance(_metadata, dict)
        inherent_metadata = {
            k: v for k, v in _metadata.items() if k not in components["metadata"]
        }
        published_metadata = {
            k: v for k, v in _metadata.items() if k in components["metadata"]
        }
        assert published_metadata == components["metadata"]
        assert set(inherent_metadata.keys()) == set(Metadata(**{}).dict().keys())
    else:
        raise NotImplementedError(component)


def _failer(
    thing_store: ThingStore, file_id: str, component: str, base_path: str
) -> None:
    """Intelligently attempt to load and test a failed 'thing'"""
    if component == "dataset":
        dataset = thing_store.get_dataset(file_identifier=file_id)
        assert dataset is None
    elif component == "parameters":
        params = thing_store.get_parameters(file_identifier=file_id)
        assert params == {}
    elif component == "metrics":
        metrics = thing_store.get_metrics(file_identifier=file_id)
        assert metrics == {}
    elif component == "artifacts_folder":
        thing_store.get_artifacts(file_identifier=file_id, target_path=base_path)
        assert not os.path.exists(f"{base_path}/artifacts")
    elif component == "metadata":
        _metadata = thing_store.get_metadata(file_identifier=file_id)
        # TODO: Test dynamic elements more effectively.
        dynamic_elements = {
            k: v
            for k, v in _metadata.items()
            if k.startswith("TS_HAS") or k == "DATASET_DATE"
        }
        assert {k: v for k, v in _metadata.items() if k not in dynamic_elements} == {
            "FILE_ID": file_id,
            "FILE_VERSION": 1,
            "DATASET_VALID": True,
        }
        assert isinstance(_metadata["DATASET_DATE"], datetime)
        assert isinstance(_metadata["TS_HAS_DATASET"], bool)
        assert isinstance(_metadata["TS_HAS_PARAMETERS"], bool)
        assert isinstance(_metadata["TS_HAS_METADATA"], bool)
        assert isinstance(_metadata["TS_HAS_ARTIFACTS"], bool)
        assert isinstance(_metadata["TS_HAS_METRICS"], bool)
        assert isinstance(_metadata["TS_HAS_FUNCTION"], bool)
        assert isinstance(_metadata["TS_HAS_EMBEDDING"], bool)
    else:
        raise NotImplementedError(component)


@pytest.mark.usefixtures(
    "remote_thing_store",
    "component_combination",
    "testing_artifacts_folder",
    "test_temporary_folder",
)
def test_component_logging(
    remote_thing_store,
    component_combination,
    testing_artifacts_folder,
    test_temporary_folder,
):
    """Save component combinations to Thing Store"""
    # Get the parameters
    params = {k: v for k, v in components.items() if k in component_combination}
    if "artifacts_folder" in params:
        params.update(artifacts_folder=testing_artifacts_folder)
    # Try to log this job and get a file id.
    try:
        file_id = remote_thing_store.log(**params)
    except BaseException as e:
        raise Exception(f"Broke attempting to log {component_combination}") from e
    test_path = os.path.join(test_temporary_folder, "test_component_logging", file_id)
    os.makedirs(test_path)
    # Here we're going to run the gamut of loading saved things.
    # We're going to validate that calling the others results in failure.
    for component in components:
        if component in params:  # I *should* be able to get it.
            try:
                _loader(
                    thing_store=remote_thing_store,
                    file_id=file_id,
                    component=component,
                    base_path=test_path,
                )
            except BaseException as e:
                raise Exception(component, components[component]) from e
        else:  # I should *not* be able to get it.
            _failer(
                thing_store=remote_thing_store,
                file_id=file_id,
                component=component,
                base_path=test_path,
            )
