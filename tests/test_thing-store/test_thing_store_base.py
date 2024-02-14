"""Test base thing store object."""
import pyarrow as pa
import pytest
from thethingstore.thing_store_base import ThingStore
from pyarrow.fs import FileSystem, LocalFileSystem, S3FileSystem


test_cases = [
    (LocalFileSystem(),),
    (S3FileSystem(),),
]


@pytest.mark.usefixtures("test_temporary_folder")
@pytest.mark.parametrize(("metadata_filesystem",), test_cases)
def test_base_thing_store_object(
    metadata_filesystem: FileSystem, test_temporary_folder: str
):
    """Test the base thing store object.

    This is relatively easy! None of the methods should be implemented.
    """
    local_folder = f"{test_temporary_folder}-{metadata_filesystem.type_name}"
    my_useless_thing_store = ThingStore(
        metadata_filesystem=metadata_filesystem, local_storage_folder=local_folder
    )
    # Checking the init.
    # The tempdir should be None.
    assert my_useless_thing_store._tempdir is None
    # The _local_storage_folder should be local_folder
    assert my_useless_thing_store._local_storage_folder == local_folder
    # The _local_fs should be an instance of LocalFileSystem
    assert isinstance(my_useless_thing_store._local_fs, LocalFileSystem)
    # The _metadata_fs should be metadata_filesystem.
    assert my_useless_thing_store._metadata_fs == metadata_filesystem
    # Now check the methods
    with pytest.raises(NotImplementedError):
        my_useless_thing_store.load(
            file_identifier="Does Not Matter", output_type="Does Not Matter"
        )
    with pytest.raises(NotImplementedError):
        my_useless_thing_store.log(
            dataset=pa.Table.from_pylist([{"stupid": "things"}]),
            parameters=None,
            metadata=None,
            metrics=None,
            artifacts_folder=None,
        )
    with pytest.raises(NotImplementedError):
        my_useless_thing_store.browse(thing="DoesNotMatter")
    with pytest.raises(NotImplementedError):
        my_useless_thing_store.get_artifact(
            file_identifier="DoesNotMatter",
            artifact_identifier="DoesNotMatter",
            target_path="DoesNotMatter",
        )
