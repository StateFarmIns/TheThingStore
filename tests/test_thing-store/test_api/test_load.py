import geopandas as gp
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from thethingstore.api import load
from thethingstore.api import error as tsle
from pyarrow.fs import S3FileSystem


####################################################################
#                  Test Data Identification                        #
# ---------------------------------------------------------------- #
# This utility is used determine exactly what type the incoming    #
#   file is.                                                       #
####################################################################


_id = "RANDOMTHINGYTHING"
_fl = "tests/test_data/sample.parquet"
_shp = "tests/test_data/sample.shp"
_dataset = ds.dataset(_fl)
_tbl = _dataset.to_table()
_pds = _tbl.to_pandas()

test_cases = [
    (_id, ([_id], "fileid")),
    ([_id], ([_id], "fileid")),
    ("silly_thing.parquet", (["silly_thing.parquet"], "parquet")),
    (["silly_thing.parquet"], (["silly_thing.parquet"], "parquet")),
    (_dataset, ([_dataset], "dataset")),
    ([_dataset, _dataset], ([_dataset, _dataset], "dataset")),
    (_tbl, ([_tbl], "table")),
    ([_tbl, _tbl], ([_tbl, _tbl], "table")),
    (_pds, ([_pds], "pandas")),
    ([_pds, _pds], ([_pds, _pds], "pandas")),
    ("shape.shp", (["shape.shp"], "shape")),
    (["shape.shp", "shape.shp"], (["shape.shp", "shape.shp"], "shape")),
]


@pytest.mark.parametrize(("dataset_or_filepaths", "expected_output"), test_cases)
def test__get_info_success(dataset_or_filepaths, expected_output):
    """Test _get_info successful case

    pass _get_info() a dataset/filepath and verify correct info is returned

    """
    assert load._get_info(dataset_or_filepaths) == expected_output


def test__get_info_fail():
    """Test _get_info error case

    1. pass _get_info() both a filepath and dataset simultaneously, and verify
      \"Loading multiple types\" error is raised
    2. pass _get_info() paths with different file extensions simultaneously,
      and verify \"Loading multiple file ext\" error is raised
    3. pass _get_info() a path with a bad suffix, and verify \"Cannot understand
      dataset with suffix\" error is raised
    4. pass _get_info() a dataset of an invalid type, and verify
      \"Cannot understand dataset type\" error is raised

    """
    with pytest.raises(tsle.ThingStoreLoadingError, match="Loading multiple types"):
        load._get_info(["sillything.parquet", pd.DataFrame()])
    with pytest.raises(tsle.ThingStoreLoadingError, match="Loading multiple file ext"):
        load._get_info(["sillything.parquet", "sillything.shp"])
    with pytest.raises(tsle.ThingStoreLoadingError, match="suffix .stupid"):
        load._get_info(["sillything.stupid"])
    with pytest.raises(tsle.ThingStoreLoadingError, match="type <class 'int'>"):
        load._get_info([1])


####################################################################
#                  Test The String Handler                         #
# ---------------------------------------------------------------- #
# This utility is used to ensure that filesystems get appropriate  #
#   URIs.                                                          #
####################################################################


test_cases = [
    ("tests/test_data", None, "tests/test_data"),
    ("tests/test_data", 1, "tests/test_data"),
    (1, 1, 1),
    (1, S3FileSystem(), 1),
    ("tests/test_data", S3FileSystem(), "tests/test_data"),
    ("s3://tests/test_data", S3FileSystem(), "tests/test_data"),
]


@pytest.mark.parametrize(("s3path", "filesystem", "expected_output"), test_cases)
def test__s3_str_handler(s3path, filesystem, expected_output):
    """test s3 str handler utility

    pass _s3_str_handler() an s3path and S3FileSystem, and verify output is as expected

    """
    assert load._s3_str_handler(s3path, filesystem) == expected_output


####################################################################
#              Test The Individual Load Routines                   #
# ---------------------------------------------------------------- #
# This section tests the different things that can be loaded in a  #
#   variety of ways.                                               #
####################################################################


test_cases = [
    (_fl),  # Test case one, not a list.
    ([_fl]),  # Test case two, a list.
    ([_fl, _fl]),  # Test case three, a longer list.
]


@pytest.mark.parametrize(("dataset_or_filepaths"), test_cases)
def test__load_dataset_success(dataset_or_filepaths):
    """test _load_dataset() utility for successful cases

    pass _load_dataset() a dataset/filepath, and verify output is as expected

    """
    loaded = load._load_dataset(dataset_or_filepaths=dataset_or_filepaths)
    assert [isinstance(_, ds.Dataset) for _ in loaded]
    assert [_.to_table().to_pandas().shape == (10, 5) for _ in loaded]
    assert all([_.to_table().to_pandas().equals(_pds) for _ in loaded])


def test__load_dataset_fail():
    """test _load dataset() in the case of an error

    pass _load_dataset() an item that is neither a pyarrow dataset nor a str, and verify correct error is raised

    """
    with pytest.raises(tsle.ThingStoreLoadingError, match="into dataset"):
        load._load_dataset(1)


test_cases = [
    (_shp),  # Test case one, not a list.
    ([_shp]),  # Test case two, a list.
    (gp.read_file(_shp)),  # Test case three, a GeoDF.
]


@pytest.mark.parametrize(("dataset_or_filepaths"), test_cases)
def test__load_shapefile_success(dataset_or_filepaths):
    """test _load_shapefile() utility for successful cases

    pass _load_shapefile() a dataset/filepath, and verify output is as expected

    """
    loaded = load._load_shape(dataset_or_filepaths, {})
    assert [isinstance(_, gp.GeoDataFrame) for _ in loaded]
    assert [_.shape == (1, 1) for _ in loaded]
    assert [_.equals(gp.read_file(_shp)) for _ in loaded]


def test__load_shapefile_fail():
    """test _load_shapefile() in the case of an error

    pass _load_dataset() an item that is neither a geopandas dataframe nor a str, and verify correct error is raised

    """
    with pytest.raises(tsle.ThingStoreLoadingError, match="into GeoPandas"):
        load._load_shape(1)


test_cases = [
    (ds.dataset(_fl)),  # Test case one, not a list.
    ([ds.dataset(_fl)]),  # Test case two, a list.
    (ds.dataset(_fl).to_table()),  # Test case three, a table.
    (
        [ds.dataset(_fl).to_table(), ds.dataset(_fl).to_table()]
    ),  # Test case four, a list of tables.
]


@pytest.mark.parametrize(("dataset_or_filepaths"), test_cases)
def test__load_table_success(dataset_or_filepaths):
    """test _load_table() utility for successful cases

    pass _load_table() a dataset/filepath, and verify output is as expected

    """
    loaded = load._load_table(dataset_or_filepaths, {})
    assert [isinstance(_, pa.Table) for _ in loaded]
    assert [_.to_pandas().shape == (10, 5) for _ in loaded]
    assert [_ == _tbl for _ in loaded]


def test__load_table_fail():
    """test _load_table() in the case of an error

    pass _load_table() an item that is neither a pyarrow dataset nor a pyarrow table, and verify correct error is raised

    """
    with pytest.raises(tsle.ThingStoreLoadingError, match="into Table"):
        load._load_table(1)


test_cases = [
    (_tbl),  # Test case one, not a list.
    ([_tbl]),  # Test case two, a list.
    (_tbl.to_pandas()),  # Test case three, a table.
    ([_tbl.to_pandas(), _tbl.to_pandas()]),  # Test case four, a list of tables.
]


@pytest.mark.parametrize(("dataset_or_filepaths"), test_cases)
def test__load_pandas_success(
    dataset_or_filepaths,
):
    """test _load_pandas() utility for successful cases

    pass _load_pandas() a dataset/filepath, and verify output is as expected

    """
    loaded = load._load_pandas(dataset_or_filepaths, {})
    assert [isinstance(_, pd.DataFrame) for _ in loaded]
    assert [_.shape == (10, 5) for _ in loaded]
    assert [_.equals(_pds) for _ in loaded]


def test__load_pandas_fail():
    """test _load_pandas() in the case of an error

    pass _load_pandas() an item that is neither a pyarrow table
    nor a pandas dataframe, and verify correct error is raised

    """
    with pytest.raises(tsle.ThingStoreLoadingError, match="into Pandas"):
        load._load_pandas(1)


####################################################################
#                   Test The Load Mapping                          #
# ---------------------------------------------------------------- #
# This section tests the different things that can be loaded by    #
#   the loader in a variety of ways.                               #
####################################################################


input_mapping = {
    "parquet": _fl,
    "dataset": _dataset,
    "table": _tbl,
    "pandas": _pds,
    # 'Shape': ['list of two']
}

output_mapping_singular = {
    "parquet": _pds,
    "dataset": _pds,
    "table": _pds,
    "pandas": _pds,
}

_dbl_pds = pd.concat([_pds, _pds])
output_mapping_plural = {
    "parquet": _dbl_pds,
    "dataset": _dbl_pds,
    "table": _dbl_pds,
    "pandas": _dbl_pds,
}

test_cases = []
# Two sets of cases - one for loading and one for passthrough.
# This is for loading.
for input_type, input_var in input_mapping.items():
    _input_type = input_type
    # Here we're going to ensure that the filesystem can read
    while True:
        try:
            output_type = load._next_mapping[_input_type]
        except KeyError:
            break
        # Case where we load one thing.
        test_cases.append(
            [
                input_var,  # dataset_or_filepaths
                input_type,  # dataset_type,
                output_type,  # output_format
                output_mapping_singular[output_type],  # expected_output
            ]
        )
        # Case where we load two things.
        test_cases.append(
            [
                [input_var, input_var],  # dataset_or_filepaths
                input_type,  # dataset_type,
                output_type,  # output_format
                output_mapping_plural[output_type],  # expected_output
            ]
        )
        _input_type = output_type


@pytest.mark.parametrize(
    (
        "dataset_or_filepaths",
        "dataset_type",
        "output_format",
        "expected_output",
    ),
    test_cases,
)
def test__map_load_success(
    dataset_or_filepaths,
    dataset_type,
    output_format,
    expected_output,
):
    """test _map_load() for successful case

    pass _map_load() a dataset, dataset_type, output_format for _load_dataset
        verify output is as expected

    """
    try:
        actual_output, _ = load._map_load(
            dataset_or_filepaths=dataset_or_filepaths,
            dataset_type=dataset_type,
            output_format=output_format,
        )
        if output_format == "parquet":
            assert actual_output == expected_output
            return
        elif output_format == "dataset":
            test_frame = pa.concat_tables(
                [_.to_table() for _ in actual_output]
            ).to_pandas()
        elif output_format == "table":
            test_frame = pd.concat([_.to_pandas() for _ in actual_output])
        elif output_format == "pandas":
            test_frame = pd.concat(actual_output)
        else:
            raise Exception("Not accounted for in test suite.")
        pd.testing.assert_frame_equal(test_frame, _pds)
    except BaseException as e:
        raise Exception(
            f"""
        Load Error:

        dataset_or_filepaths: {dataset_or_filepaths}
        dataset_type: {dataset_type}
        output_format: {output_format}
        """
        ) from e


# def test__map_load_fail(
#     dataset_or_filepaths,
#     dataset_type,
#     output_format,
#     metadata_bucket,
#     metadata_prefix,
#     load_dataset_kwargs,
# ):
#     """test _map_load() in case of error"""
#     raise NotImplementedError
# TODO: Let me think on this.
