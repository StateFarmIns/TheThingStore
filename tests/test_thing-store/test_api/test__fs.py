"""Test filesystem utilities."""

import os
import pytest
from thethingstore.api._fs import get_fs, ls_dir
from pyarrow.fs import FileSystem, LocalFileSystem, S3FileSystem, copy_files

test_cases = [
    # (
    #     "s3://tmp/prefix",
    #     "tmp/prefix",
    #     S3FileSystem,
    # ),  # TODO: Figure out why from_uri breaks in gitlab - works locally
    ("file:///local/absolute/path", "/local/absolute/path", LocalFileSystem),
    ("Stupid/Example", "Stupid/Example", LocalFileSystem),
    (
        ["s3://thingy/thingy", "file:///shwatever"],
        ["s3://thingy/thingy", "file:///shwatever"],
        LocalFileSystem,
    ),
    (
        {"path": "anything", "thing": "whateverandeveramen"},
        {"path": "anything", "thing": "whateverandeveramen"},
        LocalFileSystem,
    ),
]


@pytest.mark.usefixtures("moto_server")
@pytest.mark.parametrize(("flpath", "expected_path", "expected_fs"), test_cases)
def test_get_fs(flpath: str, expected_path: str, expected_fs: FileSystem):
    """Test get_fs."""
    actual_path, actual_fs = get_fs(flpath)
    assert isinstance(actual_fs, expected_fs)
    assert actual_path == expected_path


def test_get_fs_error():
    """Test get_fs error."""
    with pytest.raises(NotImplementedError):
        get_fs({"thing"})


@pytest.mark.usefixtures(
    "testing_artifacts_folder", "moto_server", "test_temporary_folder"
)
def test_ls_dir(testing_artifacts_folder, test_temporary_folder):
    """Test ls_dir."""

    def run_test(filesystem, target_folder):
        # Here we write out a structure
        filesystem.create_dir(target_folder)
        copy_files(
            source=testing_artifacts_folder,
            destination=target_folder,
            source_filesystem=LocalFileSystem(),
            destination_filesystem=filesystem,
        )
        # Check single file.
        actual_paths = ls_dir(f"{target_folder}/numpy.npy", filesystem=filesystem)
        # raise Exception(actual_paths)
        actual_paths = actual_paths.replace(f"{target_folder}/", "")
        assert actual_paths == "numpy.npy"
        # Check shallow folder.
        actual_paths = ls_dir(target_folder, filesystem=filesystem)
        actual_paths = [_.replace(f"{target_folder}/", "") for _ in actual_paths]
        assert set(actual_paths) == set(["numpy.npy", "list.txt"])
        # Check deep folder
        filesystem.create_dir(f"{target_folder}/deeper")
        copy_files(
            source=testing_artifacts_folder,
            destination=f"{target_folder}/deeper",
            source_filesystem=LocalFileSystem(),
            destination_filesystem=filesystem,
        )
        actual_paths = ls_dir(target_folder, filesystem=filesystem)
        actual_paths = [_.replace(f"{target_folder}/", "") for _ in actual_paths]
        assert set(actual_paths) == set(["numpy.npy", "list.txt", "deeper"])

    local_file_system = LocalFileSystem()
    s3_file_system = S3FileSystem(
        endpoint_override="http://localhost:5000", allow_bucket_creation=True
    )
    run_test(local_file_system, os.path.join(test_temporary_folder, "local_ls_dir"))
    run_test(s3_file_system, "silly/testing/folder")
