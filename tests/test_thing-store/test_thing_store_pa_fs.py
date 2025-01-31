"""Test unit functionality.

Note that currently this is just mainly validating pyarrow functionality.
"""

import numpy as np
import os
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest
import uuid
from thethingstore.api.save import save
from thethingstore.api.load import materialize
from thethingstore.thing_store_pa_fs import (
    get_user,
    pyarrow_tree,
    create_default_dataset,
    recursive_make_dir,
    FileSystemThingStore,
    _lock_metadata,
    _unlock_metadata,
    _metadata_lock_status,
)
from pyarrow.fs import LocalFileSystem, S3FileSystem, copy_files
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import BallTree
from typing import Any, Dict, Union

test_filesystems = [  # nosec: These are NOT security violations.
    (LocalFileSystem, dict()),  # This is a local filesystem.
    (
        S3FileSystem,
        dict(
            endpoint_override="http://localhost:5000",
            # allow_bucket_creation=True,
            access_key="testing",  # nosec
            secret_key="testing",  # nosec
            session_token="testing",  # nosec
        ),
    ),  # This is a remote filesystem.
]


def write_files(
    fs: Union[LocalFileSystem, S3FileSystem], local_folder: str, output_path: str
) -> None:
    """Copy test data."""
    test_unique_id = uuid.uuid1().hex
    path = os.path.join(output_path, "test_copy_files", test_unique_id)
    if isinstance(fs, S3FileSystem):
        # This strips a preceding /
        path = path[1:]
        fs.create_dir(path)
    elif isinstance(fs, LocalFileSystem):
        os.makedirs(path)
    try:
        copy_files(
            source=local_folder,
            destination=path,
            destination_filesystem=fs,
        )
    except BaseException as e:  # noqa: B036 - Catchall
        raise Exception(path) from e
    return path


@pytest.fixture(params=test_filesystems, ids=lambda x: x[0])
def source_filesystem(
    request, testing_artifacts_folder, test_temporary_folder, moto_server, _set_env
):
    """Return source filesystem."""
    fs = request.param[0](**request.param[1])
    path = write_files(fs, testing_artifacts_folder, test_temporary_folder)
    return fs, path


@pytest.fixture(params=test_filesystems, ids=lambda x: x[0])
def target_filesystem(
    request, testing_artifacts_folder, test_temporary_folder, moto_server, _set_env
):
    """Return target filesystem."""
    fs = request.param[0](**request.param[1])
    path = write_files(fs, testing_artifacts_folder, test_temporary_folder)
    return fs, path


@pytest.mark.usefixtures("source_filesystem", "target_filesystem")
def test_file_copy(source_filesystem, target_filesystem):
    """Test copy."""
    source_fs, source_path = source_filesystem
    target_fs, target_path = target_filesystem
    copy_files(
        source=source_path,
        source_filesystem=source_fs,
        destination=target_path,
        destination_filesystem=target_fs,
    )
    left = pyarrow_tree(source_path, source_fs, file_info=False)
    right = pyarrow_tree(target_path, target_fs, file_info=False)
    assert left == right


####################################################################
#             Unit Tests for PyArrow Metadata Functions            #
####################################################################

test_cases = [
    ("file:///opt/ml/metadata/resource-metadata.json", "USER_UNKNOWN_NO_METADATA"),
    ("tests/test_data/metadata.json", "STEVE!"),
]


@pytest.mark.parametrize(("metadata_path", "expected_user"), test_cases)
def test_get_user(metadata_path: str, expected_user: str) -> None:
    """Test get user."""
    assert get_user(metadata_path=metadata_path) == expected_user


@pytest.mark.usefixtures("source_filesystem")
def test_create_default_dataset(source_filesystem):
    """Test default dataset creation."""
    fs, path = source_filesystem
    tgt_prefix = "test_create_default_dataset"
    new_path = os.path.join(path, tgt_prefix, "silly")
    # 1. No file present.
    assert fs.get_file_info(os.path.join(new_path, "test1.parquet")).type.value == 0
    create_default_dataset(
        filesystem=fs,
        path=os.path.join(new_path, "test1.parquet"),
        schema=pa.schema({"test": "int64"}),
    )
    # 2. File now present.
    assert fs.get_file_info(os.path.join(new_path, "test1.parquet")).type.value != 0
    _data = ds.dataset(os.path.join(new_path, "test1.parquet"), filesystem=fs)
    assert _data
    assert _data.schema == pa.schema({"test": "int64"})
    assert _data.count_rows() == 0
    # 3. Pre-existing file.
    pq.write_table(
        pa.Table.from_pylist(
            [{"silly": 1, "things": 2.0}],
            schema=pa.schema({"silly": "float", "things": "float"}),
        ),
        where=os.path.join(new_path, "test2.parquet"),
        filesystem=fs,
    )
    create_default_dataset(
        filesystem=fs,
        path=os.path.join(new_path, "test2.parquet"),
        schema=pa.schema({"test": "int64"}),
    )
    # 4. Original file unchanged
    assert fs.get_file_info(os.path.join(new_path, "test2.parquet")).type.value != 0
    _data = ds.dataset(os.path.join(new_path, "test2.parquet"), filesystem=fs)
    assert _data
    assert _data.schema == pa.schema({"silly": "float", "things": "float"})
    assert _data.count_rows() == 1
    assert _data.to_table().to_pylist() == [{"silly": 1, "things": 2.0}]


@pytest.fixture
def filesystem_metadata(source_filesystem):
    """Provide filesystem thingstore."""
    fs, path = source_filesystem
    tgt_prefix = "test_ts_pa"
    new_path = os.path.join(path, tgt_prefix)
    return new_path, FileSystemThingStore(
        metadata_filesystem=fs,
        managed_location=new_path,
    )


@pytest.mark.usefixtures("filesystem_metadata")
def test_metadata_locking(filesystem_metadata):
    """Test locking."""
    new_path, ts = filesystem_metadata
    assert (
        not ds.dataset(
            os.path.join(new_path, "metadata-lock.parquet"),
            filesystem=ts._metadata_fs,
        )
        .to_table()
        .to_pylist()
    )
    assert not _metadata_lock_status(ts)
    _lock_metadata(ts)
    assert (
        ds.dataset(
            os.path.join(new_path, "metadata-lock.parquet"),
            filesystem=ts._metadata_fs,
        )
        .to_table()
        .to_pylist()[0]["USER"]
        == "USER_UNKNOWN_NO_METADATA"
    )
    assert _metadata_lock_status(ts)
    _unlock_metadata(ts)
    assert (
        not ds.dataset(
            os.path.join(new_path, "metadata-lock.parquet"),
            filesystem=ts._metadata_fs,
        )
        .to_table()
        .to_pylist()
    )
    assert not _metadata_lock_status(ts)


test_cases = [
    (
        {"onedir": ["two.thing", "three.thing"]},
        {"onedir": {"three.thing": "file", "two.thing": "file"}},
    ),
    (
        {"onedir": {"twodir": ["one.thing"], "threedir": ["two.thing"]}},
        {
            "onedir": {
                "threedir": {"two.thing": "file"},
                "twodir": {"one.thing": "file"},
            }
        },
    ),
    (
        {
            "onedir": {
                "twodir": [],
                "threedir": {"fourdir": ["one.thing"], "fivedir": {"sixdir": []}},
            }
        },
        {
            "onedir": {
                "threedir": {
                    "fivedir": {"sixdir": {}},
                    "fourdir": {"one.thing": "file"},
                },
                "twodir": {},
            }
        },
    ),
    (
        {"onedir": ["a.thing"]},
        {"onedir": {"a.thing": "file"}},
    ),
]  # Represents deeply nested file structure.


def _mkdir(path: str, dirs: Dict[str, Any], filesystem) -> None:
    """Make directories."""
    for k, v in dirs.items():
        # The keys will always be a folder
        filesystem.create_dir(os.path.join(path, k))
        if isinstance(v, list):
            # Make files
            for _ in v:
                with filesystem.open_output_stream(os.path.join(path, k, _)) as f:
                    f.write(b"Nothing")
        elif isinstance(v, dict):
            # Make a folder.
            _mkdir(os.path.join(path, k), v, filesystem=filesystem)


@pytest.mark.parametrize(("folder_structure", "expectation"), test_cases)
@pytest.mark.usefixtures("filesystem_metadata")
def test_pyarrow_tree(
    folder_structure, expectation, filesystem_metadata, request
) -> Any:
    """Test pyarrow_tree."""
    new_path, ts = filesystem_metadata
    test_id = request.node.callspec.id
    tgt_path = os.path.join(new_path, "test_pyarrow_tree", test_id)
    ts._metadata_fs.create_dir(tgt_path, recursive=True)
    _mkdir(tgt_path, folder_structure, filesystem=ts._metadata_fs)
    assert (
        pyarrow_tree(tgt_path, filesystem=ts._metadata_fs, max_depth=6, file_info=False)
        == expectation
    )


@pytest.mark.usefixtures("test_temporary_folder")
def test_pyarrow_tree_single_file(test_temporary_folder):
    """Test what happens when pyarrow tree is called on a file."""
    _path = f"{test_temporary_folder}/test_pyarrow_tree_single_file"
    os.makedirs(_path)
    with open(f"{_path}/stupidthing.thing", "w") as f:
        f.write("CeaseYourMindlessChatter!")
    with pytest.raises(NotADirectoryError, match="Not a directory"):
        pyarrow_tree(f"{_path}/stupidthing.thing", filesystem=LocalFileSystem())


@pytest.mark.usefixtures("filesystem_metadata")
def test_pyarrow_tree_edge_cases(filesystem_metadata, request) -> Any:
    """Test edge cases."""
    new_path, ts = filesystem_metadata
    test_id = request.node.callspec.id
    tgt_path = os.path.join(new_path, "test_pyarrow_tree_edge_cases", test_id)
    ts._metadata_fs.create_dir(tgt_path, recursive=True)
    _mkdir(tgt_path, test_cases[2][0], filesystem=ts._metadata_fs)
    assert pyarrow_tree(
        tgt_path, filesystem=ts._metadata_fs, max_depth=2, file_info=False
    ) == {"onedir": {"threedir": "...", "twodir": "..."}}


####################################################################
#                    Test Item Save and Load                       #
# ---------------------------------------------------------------- #
# This section conducts testing for saving and loading items over  #
#   the different filesystems.                                     #
####################################################################

rng = np.random.RandomState(0)
sample = rng.random_sample((10, 3))


X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

deeply_nested_dict = {
    "key01": "VALUE_01",
    "key02": "VALUE_02",
    "key03_nested_dict": {
        "key03A": "VALUE_03A",
        "key03B": {"key03BA": 16, "key03BB": 20},
        "key03C": {"key03CA": 16, "key03CB": 20},
    },
}

other_deeply_nested_dict = {
    "key": [
        {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3",
            "key4": "value4",
            "key5": "value5",
            "key6_nested": {
                "key6A": "value6A",
                "key6B": "value6B",
                "key6C": [
                    {
                        "key6CA": "value6CA",
                        "key6CB": "value6CB",
                        "key6CC": {
                            "key6CCA": [
                                {
                                    "key6CCAA": 1,
                                    "key6CCAB": 2,
                                    "key6CCAC": 3,
                                    "key6CCAD": True,
                                },
                                {
                                    "key6CCAA": -0.5,
                                    "key6CCAB": 10,
                                    "key6CCAC": 3,
                                    "key6CCAD": False,
                                },
                                {
                                    "key6CCAA": -0.3,
                                    "key6CCAB": 1,
                                    "key6CCAC": 2,
                                    "key6CCAD": False,
                                },
                                {
                                    "key6CCAA": 0.05,
                                    "key6CCAB": 0.03,
                                    "key6CCAC": 0.025,
                                    "key6CCAD": False,
                                },
                                {
                                    "key6CCAA": 0.432,
                                    "key6CCAB": 5000.1,
                                    "key6CCAC": 0.234,
                                    "key6CCAD": True,
                                },
                            ],
                        },
                        "key6CD": "value6CD",
                        "key6CE": {
                            "key6CEA": 0.0,
                            "key6CEB": 999.99,
                        },
                        "key6CF": "PROPOSED",
                        "key6CG": {
                            "key6CGA": {
                                "key6CGAA": "value6CGAA",
                                "key6CGAB": "value6CGAA",
                                "key6CGAC": "value6CGAA",
                                "key6CGAD": "value6CGAA",
                            },
                            "key6CGB": [
                                "value6CGBA",
                                "value6CGBB",
                                "value6CGBC",
                                "value6CGBD",
                                "value6CGBE",
                                "value6CGBF",
                                "value6CGBG",
                                "value6CGBH",
                            ],
                        },
                        "key6CH": {"key6CHA": "value6CHA"},
                        "key6CI": {},
                    }
                ],
                "key6D": None,
                "key6E": "value6E",
            },
        }
    ],
}

items = {
    "single_int": (1, {"ts-atomic-item.parquet": "file"}),
    "single_float": (1.0, {"ts-atomic-item.parquet": "file"}),
    "single_str": ("1", {"ts-atomic-item.parquet": "file"}),
    "dict_ints": ({"a": 1, "b": 2}, {"ts-dict-item.parquet": "file"}),
    "dict_mixed": ({"1": 1, "2": "b"}, {"ts-dict-item.parquet": "file"}),
    "list_ints": ([1, 2, 3], {"ts-list-item.parquet": "file"}),
    "list_mixed": ([1.0, 2.0, 3.0], {"ts-list-item.parquet": "file"}),
    "pandas_series_ints": (
        pd.Series({"a": 1, "b": 2}),
        {"ts-series-item.parquet": "file"},
    ),
    "pandas_series_floats": (
        pd.Series([1.0, 2.0, 3.0]),
        {"ts-series-item.parquet": "file"},
    ),
    "numpy_ints": (np.array([1, 2]), {"ts-numpy-item.npy": "file"}),
    "numpy_floats": (np.array([1.0, 2.0]), {"ts-numpy-item.npy": "file"}),
    "pandas_dataframe": (
        pd.DataFrame([{"stupid": "example"}]),
        {"ts-dataset-item.parquet": "file"},
    ),
    "pyarrow_array": (pa.array([1, 2, 3]), {"ts-pa-item.parquet": "file"}),
    "pyarrow_table": (
        pa.Table.from_pylist([{"stupid": "example"}]),
        {"ts-dataset-item.parquet": "file"},
    ),
    "sklearn_tree": (BallTree(sample, leaf_size=2), {"ts-skmodel-item.joblib": "file"}),
    "sklearn_model": (clf, {"ts-skmodel-item.joblib": "file"}),
    "nested_deeply_nested_dict": (
        deeply_nested_dict,
        {
            "ts-dict-item": {
                "ts-atomic-key01.parquet": "file",
                "ts-atomic-key02.parquet": "file",
                "ts-dict-key03_nested_dict": {
                    "ts-atomic-key03A.parquet": "file",
                    "ts-dict-key03B.parquet": "file",
                    "ts-dict-key03C.parquet": "file",
                },
            }
        },
    ),
    "other_deeply_nested_dict": (
        other_deeply_nested_dict,
        {
            "ts-dict-item": {
                "ts-list-key": {
                    "ts-dict-0": {
                        "ts-atomic-key2.parquet": "file",
                        "ts-atomic-key5.parquet": "file",
                        "ts-atomic-key4.parquet": "file",
                        "ts-atomic-key1.parquet": "file",
                        "ts-atomic-key3.parquet": "file",
                        "ts-dict-key6_nested": {
                            "ts-atomic-key6E.parquet": "file",
                            "ts-atomic-key6A.parquet": "file",
                            "ts-thing-key6D.pickle": "file",
                            "ts-atomic-key6B.parquet": "file",
                            "ts-list-key6C": {
                                "ts-dict-0": {
                                    "ts-atomic-key6CD.parquet": "file",
                                    "ts-dict-key6CC": {
                                        "ts-list-key6CCA": {
                                            "ts-dict-1.parquet": "file",
                                            "ts-dict-2.parquet": "file",
                                            "ts-dict-0.parquet": "file",
                                            "ts-dict-3.parquet": "file",
                                            "ts-dict-4.parquet": "file",
                                        }
                                    },
                                    "ts-dict-key6CI.parquet": "file",
                                    "ts-dict-key6CE.parquet": "file",
                                    "ts-atomic-key6CB.parquet": "file",
                                    "ts-dict-key6CH.parquet": "file",
                                    "ts-atomic-key6CF.parquet": "file",
                                    "ts-atomic-key6CA.parquet": "file",
                                    "ts-dict-key6CG": {
                                        "ts-list-key6CGB.parquet": "file",
                                        "ts-dict-key6CGA.parquet": "file",
                                    },
                                }
                            },
                        },
                    }
                }
            }
        },
    ),
}


def _ordered(obj):
    # Helper function for *deeply* nested JSON structures.
    # If this is deeply nested this function doesn't change the structure.
    # This rearranges elements during comparison.
    if isinstance(obj, dict):
        return sorted((k, _ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(_ordered(x) for x in obj)
    else:
        return obj


@pytest.mark.usefixtures("source_filesystem")
def test_update_and_delete(source_filesystem):
    """Test update and delete."""
    fs, path = source_filesystem
    tgt_prefix = "test_item_update_and_delete"
    new_path = os.path.join(path, tgt_prefix, "silly")
    fs.create_dir(new_path)
    test_ts = FileSystemThingStore(metadata_filesystem=fs, managed_location=new_path)
    _df = pd.DataFrame({"A": [1, 2, 3], "B": ["one", "two", "three"]})
    _df2 = pd.DataFrame({"X": [4, 5, 6], "Y": ["four", "five", "six"]})
    file_id = "View_TEST_" + str(0)
    _metadata = {"FILE_ID": file_id}
    # Log it into the test_mm
    test_ts.log(metadata=_metadata, dataset=_df)
    _parameter = {
        "Completion": "Incomplete.",
        "Review": "Not Reviewed",
        "Credibility": "Not Credible",
    }
    _metrics = {
        "Runtime": 1.24,
        "MemoryRequired": 234342,
        "DataSizeGB": 3.56,
    }
    _artifacts_folder = os.getcwd() + "/tests"
    test_ts._update(
        "View_TEST_0",
        dataset=_df2,
        parameters=_parameter,
        metrics=_metrics,
        artifacts_folder=_artifacts_folder,
    )
    assert test_ts.load("View_TEST_0").equals(_df2)
    assert test_ts.get_parameters("View_TEST_0") == _parameter
    assert test_ts.get_metrics("View_TEST_0") == _metrics
    assert test_ts.get_metadata("View_TEST_0")["TS_HAS_ARTIFACTS"]
    test_ts._update("View_TEST_0", parameters=None, metrics=None, artifacts_folder=None)
    assert test_ts.get_parameters("View_TEST_0")
    assert test_ts.get_metrics("View_TEST_0")
    assert test_ts.get_metadata("View_TEST_0")["TS_HAS_ARTIFACTS"]
    test_ts._delete("View_TEST_0")
    assert not test_ts.get_metadata("View_TEST_0")["DATASET_VALID"]
    test_ts._delete("View_TEST_0")
    assert "View_TEST_0" not in list(test_ts.browse()["FILE_ID"])


@pytest.mark.usefixtures("source_filesystem")
def test_recursive_make_dir(source_filesystem):
    """Test recursive make dir."""
    fs, path = source_filesystem
    tgt_prefix = "test_recursive_make_dir"
    recursive_make_dir(
        fs,
        os.path.join(path, tgt_prefix),
        "subdir1",
        "subdir2",
    )
    assert fs.get_file_info(os.path.join(path, tgt_prefix, "subdir1", "subdir2")).type


@pytest.mark.usefixtures("source_filesystem")
@pytest.mark.parametrize(("item", "expected_fileset"), items.values(), ids=items.keys())
def test_item_save_and_load(item, expected_fileset, source_filesystem):  # noqa: C901
    """Test saving and loading items in different filesystems."""
    # Test metadata
    fs, path = source_filesystem
    tgt_prefix = "test_item_save_and_load"
    new_path = os.path.join(path, tgt_prefix, "silly")
    fs.create_dir(new_path)
    # Test the *saving* of the information.
    save(item, f"{new_path}/item", filesystem=fs)
    # Test the file structure before attempting to load
    if (
        not (
            actual_struct := pyarrow_tree(
                new_path, filesystem=fs, file_info=False, max_depth=10
            )
        )
        == expected_fileset
    ):
        raise Exception(
            f"""File Structure Expectation Failure
        Expected File Structure
        -----------------------\n{expected_fileset}

        Actual File Structure
        ---------------------\n{actual_struct}
        """
        )
    # Now let's test the structure!
    try:
        loaded = materialize(new_path, filesystem=fs)[0]
    except BaseException as e:  # noqa: B036 - catchall
        raise Exception(f"Loading Failure: Unable to load {new_path}") from e
    if isinstance(item, pd.Series):
        assert item.shape == loaded.shape
        assert all(item.values == loaded.values)
    elif isinstance(item, np.ndarray):
        assert item.shape == loaded.shape
        assert np.all(item == loaded)
    elif isinstance(item, pd.DataFrame):
        assert item.equals(loaded.to_pandas())
    elif isinstance(item, BallTree):
        assert np.all(item.data == sample)
    elif isinstance(item, RandomForestClassifier):
        assert np.all(item.predict(X_test) == clf.predict(X_test))
    else:
        try:
            assert _ordered(item) == _ordered(loaded)
        except BaseException as e:  # noqa: B036 - Catchall
            raise Exception(
                f"""Test Failure

            When I tried to load an item it didn't work. Oh noes.

            Expected Item
            -------------\n{item}

            Expected Item Type
            ------------------\n{type(item)}
            Actual Item
            -----------\n{loaded}

            Actual Item Type
            ----------------\n{type(loaded)}
            """
            ) from e
