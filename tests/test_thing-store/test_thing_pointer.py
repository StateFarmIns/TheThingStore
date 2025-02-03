"""Test ThingPointer functionality."""

import uuid
import os
import pytest
import tempfile
from thethingstore.api import error as tse
from thethingstore import thing_pointer as tp
from thethingstore import FileSystemThingStore
from thethingstore.api.data_hash import dataset_digest
from pyarrow.fs import LocalFileSystem, S3FileSystem

ts_implementations = [
    (
        FileSystemThingStore,
        {
            "metadata_filesystem": LocalFileSystem(),
            "managed_location": "output",
        },
        "LocalFileSystem",
    ),
    (
        FileSystemThingStore,
        {
            "metadata_filesystem": S3FileSystem(
                endpoint_override="http://localhost:5000", allow_bucket_creation=True
            ),
            "managed_location": "output",
        },
        "S3FileSystem",
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


@pytest.fixture(params=ts_implementations, ids=lambda x: f"Data:{x[2]}")
def data_thing_store(request, test_temporary_folder):
    """Provide a Thing Store for data."""
    thing_store, params, _ = request.param
    new_params = _treat_params(params, test_temporary_folder)
    return thing_store(**new_params)


@pytest.fixture(params=ts_implementations, ids=lambda x: f"Pointer:{x[2]}")
def pointer_thing_store(request, test_temporary_folder):
    """Provide a Thing Store for pointers."""
    thing_store, params, _ = request.param
    new_params = _treat_params(params, test_temporary_folder)
    return thing_store(**new_params)


@pytest.mark.usefixtures("data_thing_store", "pointer_thing_store", "testing_data")
def test_pointer_log(data_thing_store, pointer_thing_store, testing_data):
    """Tests logging pointers."""
    # Create a data node
    dataFileID = data_thing_store.log(
        dataset=testing_data["data"],
        parameters=testing_data["parameters"],
        metadata=testing_data["metadata"],
        metrics=testing_data["metrics"],
        artifacts_folder=testing_data["artifacts"]["folder"],
    )
    # Test bump version pointer
    updateVersionPointer = data_thing_store.log(
        metadata={"FILE_ID": dataFileID}, parameters={"updated": "params"}
    )
    # Create pointer to data
    dataAddress = data_thing_store.address_of(dataFileID)
    pointerFileID = pointer_thing_store.log(
        dataset=dataAddress,
        parameters=dataAddress,
        metrics=dataAddress,
        artifacts_folder=dataAddress,
    )
    # Create a pointer chain
    pointers = []
    for _ in range(3):
        pointerAddress = pointer_thing_store.address_of(pointerFileID)
        pointerFileID = pointer_thing_store.log(
            dataset=pointerAddress,
            parameters=pointerAddress,
            metrics=pointerAddress,
            artifacts_folder=pointerAddress,
        )
        pointers.append(pointerFileID)
    # I should be able to get components from the "update version" pointer
    #   since nothing was changed, this is also the actual data
    if not data_thing_store._check_file_id(updateVersionPointer):
        raise Exception(f"The data file ID {updateVersionPointer} was not detected.")
    with tempfile.TemporaryDirectory() as t:
        data_params = data_thing_store.get_parameters(updateVersionPointer)
        data_metadata = data_thing_store.get_metadata(updateVersionPointer)
        data_dataset = data_thing_store.get_dataset(updateVersionPointer)
        data_metrics = data_thing_store.get_metrics(updateVersionPointer)
        data_artifacts_path = os.path.join(t, "data")
        data_thing_store.get_artifacts(updateVersionPointer, data_artifacts_path)
        data_artifacts = os.listdir(data_artifacts_path)
        # Get pointer components
        for _pointer in pointers:
            pointer_params = pointer_thing_store.get_parameters(_pointer)
            pointer_metadata = pointer_thing_store.get_metadata(_pointer)
            pointer_dataset = pointer_thing_store.get_dataset(_pointer)
            pointer_metrics = pointer_thing_store.get_metrics(_pointer)
            pointer_artifacts_path = os.path.join(t, "pointers")
            pointer_thing_store.get_artifacts(_pointer, pointer_artifacts_path)
            pointer_artifacts = os.listdir(pointer_artifacts_path)
            try:
                assert data_params == pointer_params, "Parameters are not equal"
                assert dataset_digest(data_dataset.to_table()) == dataset_digest(
                    pointer_dataset.to_table()
                )
                assert data_metrics == pointer_metrics, "Metrics are not equal"
                assert data_artifacts == pointer_artifacts, "Artifacts are not equal"
            except AssertionError as e:
                raise Exception(
                    f"""Thing Pointer Dereference Error: {e}
                                Parameters
                                ----------
                                Original data:\n{data_params}
                                Pointer data:\n{pointer_params}

                                Metadata
                                --------
                                Original data:\n{data_metadata}
                                Pointer data:\n{pointer_metadata}

                                Dataset
                                -------
                                Original data:\n{data_dataset}
                                Pointer data:\n{pointer_dataset}

                                Metrics
                                --------
                                Original data:\n{data_metrics}
                                Pointer data:\n{pointer_metrics}

                                Artifacts
                                ---------
                                Original data:\n{data_artifacts}
                                Pointer data:\n{pointer_artifacts}
                                """
                )


@pytest.mark.usefixtures("data_thing_store")
def test_dereference_failure(data_thing_store):
    """Test fail cases for the pointer dereference operation."""
    # Dereference a file that doesn't exist
    with pytest.raises(tse.ThingStoreFileNotFoundError):
        tp.dereference(data_thing_store, "FileIDDoesNotExist", "parameters", "1")
    # Dereference a non pointer
    data = data_thing_store.log(parameters={"random": "params"})
    with pytest.raises(tse.ThingStorePointerError):
        tp.dereference(data_thing_store, data, "parameters", "1")
    # Dereference a null pointer
    pointer = data_thing_store.log(parameters=data_thing_store.address_of(data, "1"))
    data_thing_store.delete(data)
    data_thing_store.delete(data)
    with pytest.raises(tse.ThingStoreFileNotFoundError):
        tp.dereference(data_thing_store, pointer, "parameters", "1")


@pytest.mark.usefixtures("data_thing_store")
def test_address_of(data_thing_store):
    """Test the pointer address operations and checks validity of the output."""
    # Test address creation for data
    data = data_thing_store.log(parameters={"random": "params"})
    data_address = data_thing_store.address_of(data, "1")
    new_ts = data_thing_store.address_to_ts(data_address)
    if not new_ts._check_file_id(data):
        raise Exception(f"The data file ID does not exist\n{data_address}")
    # Test again for pointers
    pointer = data_thing_store.log(parameters=data)
    pointer_address = data_thing_store.address_of(pointer, "1")
    new_ts = data_thing_store.address_to_ts(pointer_address)
    if not new_ts._check_file_id(pointer):
        raise Exception(f"The pointer file ID does not exist\n{pointer_address}")


def test_get_size_success(data_thing_store):
    """Test the get_size function reliability."""
    data = data_thing_store.log(parameters={"random": "params"})
    size_data = data_thing_store.get_size(data)
    assert isinstance(size_data, int)
    pointer = data_thing_store.log(parameters=data)
    size_pointer = data_thing_store.get_size(pointer)
    assert isinstance(size_pointer, int)
    # Sniff test for size values
    assert size_data > size_pointer


def test_get_size_failure(data_thing_store):
    """Test fail cases for the get_size function."""
    with pytest.raises(tse.ThingStoreFileNotFoundError):
        data_thing_store.get_size("ThisFileIDDoesNotExist")
