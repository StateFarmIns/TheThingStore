import pytest
import tempfile
import time
import uuid
import numpy as np
import os
import pandas as pd
import pyarrow as pa
from datetime import datetime
from pyarrow.fs import LocalFileSystem, S3FileSystem
from thethingstore.thing_store_pa_fs import FileSystemThingStore
from thethingstore.api import data_hash as tsh


datasets_to_hash = [
    (pd.DataFrame({"column1": [420, 380, 390], "column2": [50, 40, 45]})),
    (pd.DataFrame({"column1": [420, 380, 390], "column2": [50, 40, 45]})),
    (pd.DataFrame({"differentColumnName": [420, 380, 390], "column2": [50, 40, 45]})),
    (pd.DataFrame({"column1": [1, 2, 3], "column2": [50, 40, 45]})),
    (
        pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "Elf": ["Galadriel", "Legolas", "Gil-galad", "Elrond", "Thranduil"],
                    "Age": [8372, 2931, 6600, 6497, 5480],
                }
            )
        )
    ),
    (
        pa.Table.from_pandas(
            pd.DataFrame(
                {
                    "Hobbit": ["Frodo", "Samwise", "Pippin", "Meriadoc", "Bilbo"],
                    "Height": [1.24, 1.27, 1.24, 1.27, 1.07],
                }
            )
        )
    ),
    ("tests/test_data/sample.parquet"),
]

big_datasets = [
    (
        1_000,  # num_of_rows
        ["column1", "column2", "column3", "column4", "column5"],  # cols
        100,  # chunk_size
    ),
    (
        10_000,  # num_of_rows
        ["column1", "column2", "column3", "column4", "column5", "column6"],  # cols
        1000,  # chunk_size
    ),
    (
        100_000,  # num_of_rows
        ["column1", "column2", "column3", "column4", "column5", "column6"],  # cols
        1000,  # chunk_size
    ),
    (
        1_000_000,  # num_of_rows
        ["column1", "column2", "column3", "column4", "column5", "column6"],  # cols
        1000,  # chunk_size
    ),
]

thing_store_sets = [
    (
        FileSystemThingStore,
        {
            "metadata_filesystem": LocalFileSystem(),
            "managed_location": "output",
        },
        "LocalFilesystem",
    ),
    (
        FileSystemThingStore,
        {
            "metadata_filesystem": S3FileSystem(
                endpoint_override="http://localhost:5000", allow_bucket_creation=True
            ),
            "managed_location": "output",
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
    params=[_ for _ in thing_store_sets if _[2] == "LocalFilesystem"],
    ids=lambda x: f"Local:{x[2]}",
)
@pytest.mark.usefixtures("test_temporary_folder")
def local_thing_store(request, test_temporary_folder):
    # This creates a local thing_store.
    thing_store, params, test_name = request.param
    new_params = _treat_params(params, test_temporary_folder)
    yield thing_store(**new_params)


@pytest.fixture(
    params=[_ for _ in thing_store_sets if _[2] != "LocalFilesystem"],
    ids=lambda x: f"Remote:{x[2]}",
)
@pytest.mark.usefixtures("test_temporary_folder")
def remote_thing_store(request, test_temporary_folder):
    # This creates a remote thing_store.
    thing_store, params, test_name = request.param
    new_params = _treat_params(params, test_temporary_folder)
    yield thing_store(**new_params)


@pytest.mark.usefixtures(
    "remote_thing_store",
    "local_thing_store",
    "testing_data",
)
def test_dataset_digest(remote_thing_store, local_thing_store, testing_data):
    # Make the DATASET_DATE deterministic
    time = datetime(year=2000, month=1, day=1)
    testing_data["metadata"].update({"FILE_ID": "TESTFILEID", "DATASET_DATE": time})

    # Log a 'thing' to remote thing_store
    remote_thing_store.log(
        dataset=testing_data["data"],
        parameters=testing_data["parameters"],
        metadata=testing_data["metadata"],
        metrics=testing_data["metrics"],
        artifacts_folder=testing_data["artifacts"]["folder"],
    )
    # Log a 'thing' to local thing_store
    local_thing_store.log(
        dataset=testing_data["data"],
        parameters=testing_data["parameters"],
        metadata=testing_data["metadata"],
        metrics=testing_data["metrics"],
        artifacts_folder=testing_data["artifacts"]["folder"],
    )

    # Calculate each hash twice
    remote_metadata_hash_1 = remote_thing_store.get_metadata_hash()
    remote_metadata_hash_2 = remote_thing_store.get_metadata_hash()
    local_metadata_hash_1 = local_thing_store.get_metadata_hash()
    local_metadata_hash_2 = local_thing_store.get_metadata_hash()

    # Two hashes of the same dataset should be identical
    assert remote_metadata_hash_1 == remote_metadata_hash_2
    assert local_metadata_hash_1 == local_metadata_hash_2

    # Log a new 'thing' to the remote thing_store
    remote_thing_store.log(
        dataset=testing_data["data"],
        parameters=testing_data["parameters"],
        metadata=testing_data["metadata"],
        metrics=testing_data["metrics"],
        artifacts_folder=testing_data["artifacts"]["folder"],
    )
    # Log a new 'thing' to the local thing_store
    local_thing_store.log(
        dataset=testing_data["data"],
        parameters=testing_data["parameters"],
        metadata=testing_data["metadata"],
        metrics=testing_data["metrics"],
        artifacts_folder=testing_data["artifacts"]["folder"],
    )

    remote_metadata_hash_changed = remote_thing_store.get_metadata_hash()
    local_metadata_hash_changed = local_thing_store.get_metadata_hash()

    # The new hashes should be different than the old ones
    assert remote_metadata_hash_changed != remote_metadata_hash_1
    assert local_metadata_hash_changed != local_metadata_hash_1

    # Pull back the metadata, but drop the last row
    df_remote = remote_thing_store.browse()
    df_remote = df_remote[:-1]
    df_local = local_thing_store.browse()[:-1]

    # Recalculate hashes
    remote_metadata_hash_3 = tsh.dataset_digest(df_remote)
    local_metadata_hash_3 = tsh.dataset_digest(df_local)

    # A + B - B = A
    assert remote_metadata_hash_3 == remote_metadata_hash_1
    assert local_metadata_hash_3 == local_metadata_hash_1


@pytest.mark.parametrize(("dataset"), datasets_to_hash)
def test_dataset_digest_ram(dataset):
    """Test Pandas and PyArrow datasets from RAM"""
    # Calculate hashes twice
    hash_hex_1, hash_bytes_1, hash_int_1 = (
        tsh.dataset_digest(df=dataset, return_type="hex"),
        tsh.dataset_digest(df=dataset, return_type="bytes"),
        tsh.dataset_digest(df=dataset, return_type="int"),
    )
    hash_hex_2, hash_bytes_2, hash_int_2 = (
        tsh.dataset_digest(df=dataset, return_type="hex"),
        tsh.dataset_digest(df=dataset, return_type="bytes"),
        tsh.dataset_digest(df=dataset, return_type="int"),
    )
    assert hash_hex_1 == hash_hex_2
    assert hash_bytes_1 == hash_bytes_2
    assert hash_int_1 == hash_int_2


@pytest.mark.usefixtures(
    "remote_thing_store",
    "testing_data",
    "client",
)
def test_dataset_digest_s3(remote_thing_store, testing_data, client):
    # This test is for s3 only
    if not isinstance(remote_thing_store, S3FileSystem):
        return

    # Make the DATASET_DATE deterministic
    time = datetime(year=2000, month=1, day=1)
    testing_data["metadata"].update({"FILE_ID": "TESTFILEID", "DATASET_DATE": time})

    # Log a 'thing' to remote thing_store
    remote_thing_store.log(
        dataset=testing_data["data"],
        parameters=testing_data["parameters"],
        metadata=testing_data["metadata"],
        metrics=testing_data["metrics"],
        artifacts_folder=testing_data["artifacts"]["folder"],
    )

    # Get new hash of remote ts
    remote_metadata_hash_1 = tsh.dataset_digest(
        df=remote_thing_store._fs_metadata_file,
        s3_client=client,
        bucket="tmp",
        key=remote_thing_store._fs_metadata_file[4:],
        return_type="hex",
    )
    # Calculate hash again
    remote_metadata_hash_2 = tsh.dataset_digest(
        df=remote_thing_store._fs_metadata_file,
        s3_client=client,
        bucket="tmp",
        key=remote_thing_store._fs_metadata_file[4:],
        return_type="hex",
    )

    # The hashes should be the same
    assert remote_metadata_hash_1 == remote_metadata_hash_2


@pytest.fixture(scope="module")
def measure_metadata():
    with tempfile.TemporaryDirectory() as t:
        ts = FileSystemThingStore(
            metadata_filesystem=LocalFileSystem(),
            managed_location=os.path.join(t, "output"),
        )
        yield ts
        timing = ts.browse()
        timing = timing.assign(
            TIME=timing.apply(lambda x: ts.get_metrics(x.FILE_ID)["t"], axis=1)
        )[["N", "M", "Chunk Size", "TIME"]].to_markdown()
        with open("digest_measurement.md", "w") as mdfile:
            mdfile.write(timing)


@pytest.fixture(
    name="measure",
    params=big_datasets,
    ids=lambda x: f"[{x[0]}-{x[2]}]",
)
def test_dataset_digest_benchmark(measure_metadata, request):
    num_of_rows, cols, chunk_size = request.param

    # Generates deterministic data given num of rows/columns
    def generate_data(rows, columns):
        data = []
        for _ in range(rows):
            data.append([_ for _ in range(columns)])
        return data

    df = pd.DataFrame(columns=cols, dtype=np.int32)

    # Add data to dataframe in chunks to avoid ramsplosion
    for chunk_start in range(0, num_of_rows, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_of_rows)
        chunk_data = generate_data(rows=chunk_end - chunk_start, columns=len(cols))
        df_chunk = pd.DataFrame(chunk_data, columns=cols, dtype=np.int32)
        df = pd.concat([df, df_chunk], ignore_index=True)

    # Execution hash function and time execution
    start = time.time()
    hash_hex = tsh.dataset_digest(df=df)
    end = time.time()
    hash_hex_2 = tsh.dataset_digest(df=df)
    # The hash should be consistent
    assert hash_hex == hash_hex_2
    # Gather some metrics
    performance_metrics = {
        "t": end - start,
    }
    # Log to local thing_store
    return measure_metadata.log(
        dataset=df,
        metadata={
            "N": df.shape[0],
            "M": df.shape[1],
            "Chunk Size": chunk_size,
        },
        metrics=performance_metrics,
    )


def test_measurement(measure, measure_metadata):
    assert isinstance(measure, str)
