"""Test File ID utilities."""
import pytest
from thethingstore.api.error import FileIDError, ThingStoreGeneralError
from thethingstore import (
    file_id,
    FileSystemThingStore,
    MLFlowThingStore,
)
from pyarrow.fs import FileSystem, LocalFileSystem, S3FileSystem
from typing import Dict


test_cases = [
    ("fileid://stupd", {"fileid": "stupd"}),  # Simplest example
    (
        "".join(["fileid://silly.example", "/FileSystemThingStore"]),
        {"fileid": "silly.example", "implementation": "FileSystemThingStore"},
    ),  # This is how to add an implementation
    (
        "".join(
            [
                "fileid://silly.example",
                "/FileSystemThingStore",
                "?metadata_filesystem=TestA",
                "&metadata_file=TestB",
                "&metadata_lockfile=TestC",
                "&output_location=TestD",
            ]
        ),
        {
            "fileid": "silly.example",
            "implementation": "FileSystemThingStore",
            "implementation_params": {
                "metadata_filesystem": "TestA",
                "metadata_file": "TestB",
                "metadata_lockfile": "TestC",
                "output_location": "TestD",
            },
        },
    ),  # This is how to add an implementation with params
    (
        "".join(
            [
                "fileid://silly.example",
                "/FileSystemThingStore",
                "?metadata_filesystem=TestA",
                "&metadata_file=TestB",
                "&metadata_lockfile=TestC",
                "&output_location=TestD",
                "#2",
            ]
        ),
        {
            "fileid": "silly.example",
            "implementation": "FileSystemThingStore",
            "implementation_params": {
                "metadata_filesystem": "TestA",
                "metadata_file": "TestB",
                "metadata_lockfile": "TestC",
                "output_location": "TestD",
            },
            "fileversion": "2",
        },
    ),  # This is how to add an implementation with params and a fileversion
]


@pytest.mark.parametrize(("flid", "expectation"), test_cases)
def test_it(flid: str, expectation: Dict[str, str]) -> None:
    assert file_id.parse_fileid(flid) == expectation


def test_parse_fileid_failure() -> None:
    with pytest.raises(FileIDError, match="Not appropriate schema"):
        file_id.parse_fileid("ileid://stupd")
    with pytest.raises(FileIDError, match="No FileId"):
        file_id.parse_fileid("fileid://?whatever=nothing")


test_cases = [
    (
        "".join(
            [
                "fileid://silly.example",
                "/FileSystemThingStore",
                "?metadata_filesystem=LocalFileSystem",
                "&metadata_file=TestB",
                "&metadata_lockfile=TestC",
                "&output_location=TestD",
                "#2",
            ]
        ),
        {
            "fileversion": 2,
            "fileid": "silly.example",
            "implementation": FileSystemThingStore,
            "implementation_params": dict(
                metadata_filesystem=LocalFileSystem,
                metadata_file="TestB",
                metadata_lockfile="TestC",
                output_location="TestD",
            ),
        },
    ),
    (
        "".join(
            [
                "fileid://silly.example",
                "/FileSystemThingStore",
                "?metadata_filesystem=S3FileSystem",
                "&metadata_file=TestB",
                "&metadata_lockfile=TestC",
                "&output_location=TestD",
                "#avastmehearties",
            ]
        ),
        {
            "fileversion": "avastmehearties",
            "fileid": "silly.example",
            "implementation": FileSystemThingStore,
            "implementation_params": dict(
                metadata_filesystem=S3FileSystem,
                metadata_file="TestB",
                metadata_lockfile="TestC",
                output_location="TestD",
            ),
        },
    ),
    (
        "".join(
            [
                "fileid://silly.example",
                "/MLFlowThingStore",
                "?metadata_filesystem=S3FileSystem",
                "&tracking_uri=TestB",
                "&local_storage_folder=TestC",
            ]
        ),
        {
            "fileid": "silly.example",
            "fileversion": None,
            "implementation": MLFlowThingStore,
            "implementation_params": dict(
                metadata_filesystem=S3FileSystem,
                tracking_uri="TestB",
                local_storage_folder="TestC",
            ),
        },
    ),
    (
        "".join(
            [
                "fileid://silly.example",
                "/MLFlowThingStore",
                "?metadata_filesystem=S3FileSystem",
                "&tracking_uri=TestB",
                "&local_storage_folder=TestC",
            ]
        ),
        {
            "fileid": "silly.example",
            "fileversion": None,
            "implementation": MLFlowThingStore,
            "implementation_params": dict(
                metadata_filesystem=S3FileSystem,
                tracking_uri="TestB",
                local_storage_folder="TestC",
            ),
        },
    ),
]


@pytest.mark.parametrize(("flidstr", "expectation"), test_cases)
def test_data_model(flidstr: str, expectation: file_id.FileID) -> None:
    components = file_id.parse_fileid(flidstr)
    _flid = file_id.FileID(**components)
    assert isinstance(_flid, file_id.FileID)
    assert _flid.dict().keys() == expectation.keys()
    assert isinstance(_flid.implementation, type)
    assert isinstance(_flid.implementation_params["metadata_filesystem"], FileSystem)


def test_data_model_error() -> None:
    """Showcase broken examples."""
    # This is wrong because the metadata flavor isn't implemented.
    flid = "".join(
        [
            "fileid://silly.example",
            "/MistypedMetadata",
            "?metadata_filesystem=S3FileSystem",
            "&tracking_uri=TestB",
            "&local_storage_folder=TestC",
        ]
    )
    components = file_id.parse_fileid(flid)
    with pytest.raises(ThingStoreGeneralError, match="Not an accessible"):
        file_id.FileID(**components)

    # This is wrong because it doesn't HAVE an implementation
    flid = "".join(
        [
            "fileid://silly.example",
            "?metadata_filesystem=S3FileSystem",
            "&tracking_uri=TestB",
            "&local_storage_folder=TestC",
        ]
    )
    components = file_id.parse_fileid(flid)
    with pytest.raises(
        NotImplementedError, match="FileID Schema must have `implementation`"
    ):
        file_id.FileID(**components)

    # This is wrong because it doesn't have implementation arguments.
    flid = "".join(
        [
            "fileid://silly.example",
            "/MLFlowThingStore",
        ]
    )
    components = file_id.parse_fileid(flid)
    with pytest.raises(
        NotImplementedError, match="FileID Schema must have `implementation_params`"
    ):
        file_id.FileID(**components)

    # This is wrong because the metadata param fs isn't implemented.
    flid = "".join(
        [
            "fileid://silly.example",
            "/MLFlowThingStore",
            "?metadata_filesystem=SillyFileSystem",
            "&tracking_uri=TestB",
            "&local_storage_folder=TestC",
        ]
    )
    components = file_id.parse_fileid(flid)
    with pytest.raises(NotImplementedError, match="SillyFileSystem"):
        file_id.FileID(**components)
