"""Test typing utilities."""
import pyarrow.dataset as ds
import pytest
from thethingstore.types import get_info, _get_str_info, _unique
from thethingstore.api import error as tse
from typing import Any

_fl = "tests/test_data/sample.parquet"
_shp = "tests/test_data/sample.shp"
_ds = ds.dataset(_fl)
_tbl = _ds.to_table()
_df = _tbl.to_pandas()

test_cases = [
    ##############
    # Atomic Types
    ##############
    # Test a few things that are *obviously FILEID*.
    # Case insensitivity here.
    ("fileid://test", "FileID"),
    ("FileID://test", "FileID"),
    ("FileID://test", "FileID"),
    # Test a few things here that are *probably* FILEID.
    ("fileid", "FileID"),
    ("fileID", "FileID"),
    ("Longrandomstring", "FileID"),
    # Test a few things here that are *obviously* Parquet.
    (_fl, "ParquetDocument"),
    ("whatever.parquet", "ParquetDocument"),
    ("does/not/matter.parquet", "ParquetDocument"),
    ("s3://Longrandomstring.parquet", "ParquetDocument"),
    # Shape documents
    (_shp, "ShapeDocument"),
    ("whatever.shp", "ShapeDocument"),
    ("does/not/matter.shp", "ShapeDocument"),
    ("s3://Longrandomstring.shp", "ShapeDocument"),
    # Shape documents
    ("whatever.shape", "ShapeDocument"),
    ("does/not/matter.shape", "ShapeDocument"),
    ("s3://Longrandomstring.shape", "ShapeDocument"),
    # Shape documents
    ("whatever.gdb", "ShapeDocument"),
    ("does/not/matter.gdb", "ShapeDocument"),
    ("s3://Longrandomstring.gdb", "ShapeDocument"),
    # Pickle documents
    ("whatever.pickle", "PickleDocument"),
    ("does/not/matter.pickle", "PickleDocument"),
    ("s3://Longrandomstring.pickle", "PickleDocument"),
    ("whatever.pkl", "PickleDocument"),
    ("does/not/matter.pkl", "PickleDocument"),
    ("s3://Longrandomstring.pkl", "PickleDocument"),
    # JSON documents
    ("whatever.json", "JSONDocument"),
    ("does/not/matter.json", "JSONDocument"),
    # An Existing Folder
    (
        "tests/test_data/simple",
        ["ParquetDocument", "ParquetDocument", "ParquetDocument"],
    ),
]


@pytest.mark.parametrize(("obj", "expectation"), test_cases)
def test__get_str_info_success(obj: str, expectation: str) -> None:
    assert _get_str_info(obj) == expectation


def test__get_str_info_failure() -> None:
    with pytest.raises(tse.ThingStoreLoadingError, match="suffix .stupid"):
        _get_str_info("stupid.stupid")


test_cases2 = [
    # TODO: ThingStoreDataset
    # TODO: PySpark dataframe
    # PyArrow Dataset
    (_ds, "PyArrowDataset"),
    # PyArrow Table
    (_tbl, "PyArrowTable"),
    # Pandas DataFrame
    (_df, "PandasDataFrame"),
    ###################
    # Container Types #
    ###################
    # Lists
    (["onething.parquet"], ["ParquetDocument"]),
    # Note case insensitivity
    (["onething.parquet", "twothing.Parquet"], ["ParquetDocument", "ParquetDocument"]),
    # Note the recursive list.
    (
        [
            ["onething.parquet", "twothing.Parquet"],
            ["threething.parquet", "fourthing.Parquet"],
        ],
        [
            ["ParquetDocument", "ParquetDocument"],
            ["ParquetDocument", "ParquetDocument"],
        ],
    ),
    # Dictionaries
    ({"simpleexample": "howaboutthat.parquet"}, {"simpleexample": "ParquetDocument"}),
    # Mixed Dict / List
    (
        {
            "list-o-dict": [
                {"one": "thing.parquet", "two": "sillyshape.shp"},
                {"one": "thing2.parquet", "two": "sillyshape.shape"},
            ],
            "dict-o-list": {"A": ["a.shp", "b.pkl"], "B": ["c", "fileid://d"]},
        },
        {
            "list-o-dict": [
                {"one": "ParquetDocument", "two": "ShapeDocument"},
                {"one": "ParquetDocument", "two": "ShapeDocument"},
            ],
            "dict-o-list": {
                "A": ["ShapeDocument", "PickleDocument"],
                "B": ["FileID", "FileID"],
            },
        },
    ),
]


@pytest.mark.parametrize(("obj", "expectedtype"), test_cases2)
def test_get_info(obj: Any, expectedtype: Any) -> None:
    assert get_info(obj) == expectedtype


def test_get_info_fail() -> None:
    """Test get_info error case"""
    with pytest.raises(tse.ThingStoreTypeError, match="Type: <class 'int'>"):
        get_info(1)


test_cases3 = [
    # These tests demonstrate *functionality*
    ("one", {"one"}),
    (["one", "two"], {"one", "two"}),
    (["one", ["two", "three"]], {"one", "two", "three"}),
    (
        ["one", ["two", {"three_01": "three", "three_02": "four"}]],
        {"one", "two", "three", "four"},
    ),
    # These tests demonstrate *practicality*.
    (
        {"folder_01": {"file_01": "ParquetDocument", "file_02": "ShapeDocument"}},
        {"ParquetDocument", "ShapeDocument"},
    ),
    (
        {
            "partition01": {
                "a": {"file_{_}": "ParquetDocument" for _ in range(5)},
                "b": {"file_{_}": "ParquetDocument" for _ in range(5)},
            }
        },
        {"ParquetDocument"},
    ),
]


@pytest.mark.parametrize(("obj", "expectation"), test_cases3)
def test__unique(obj: Any, expectation: Any) -> None:
    """Test set compaction."""
    assert _unique(obj) == expectation


def test__unique_error() -> None:
    with pytest.raises(NotImplementedError, match="get_info"):
        _unique(1)
