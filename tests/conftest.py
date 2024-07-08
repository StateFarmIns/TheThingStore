"""Pytest Configuration File.

This file creates a temporary directory for testing and populates it
with artifacts. It also creates a fixture containing testing data to
be reused.

Finally it creates a fixture returning a single testing function that
can be used to validate if any thing store implementation functions as
expected.
"""
import boto3
import json
import numpy as np
import pandas as pd
import os
import pyarrow as pa
import pytest
import tempfile
from botocore.config import Config
from thethingstore.thing_store_base import ThingStore
from thethingstore.thing_store_elements import Metadata
from thethingstore.types import Dataset, Parameter, Metric
from moto.server import ThreadedMotoServer
from typing import Dict, List, Union

# Create some artifacts to be reused below.
artifact_table = pa.Table.from_pylist(
    [{"spiffy": "awesome", "new": "data"}, {"spiffy": "to", "new": "test"}]
)
artifact_list = ["one", "two", "three"]
artifact_np = np.array(artifact_list)


@pytest.fixture(scope="package")
def test_temporary_folder() -> str:
    """Create a fixture for housing temporary testing data.

    This fixture allows for all the testing data to be transient.
    It is created when pytest is run and is destroyed when pytest is
    done. The yield allows the temporary directory to be
    appropriately cleaned up when the testing suite is done.

    Returns
    -------
    temporary_directory: str
        Filepath for stuffing temp testing data.
    """
    with tempfile.TemporaryDirectory() as t:
        yield t


@pytest.fixture(scope="package")
def set_env() -> None:
    """Set env var"""
    os.environ["SHAPE_RESTORE_SHX"] = "YES"  # nosec
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"  # nosec
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"  # nosec
    os.environ["AWS_SECURITY_TOKEN"] = "testing"  # nosec
    os.environ["AWS_SESSION_TOKEN"] = "testing"  # nosec
    os.environ["HTTP_PROXY"] = "http://localhost:5000"  # nosec
    os.environ["HTTPS_PROXY"] = "http://localhost:5000"  # nosec


@pytest.fixture(scope="package", autouse=True)
def moto_server(set_env, test_temporary_folder):
    server = ThreadedMotoServer()
    server.start()
    client = boto3.client(
        "s3",
        endpoint_url="http://localhost:5000",
        config=Config(proxies={"https": "localhost:5000", "http": "localhost:5000"}),
    )
    # Create the bucket
    client.create_bucket(Bucket="tmp")

    yield server
    server.stop()


@pytest.fixture(scope="package")
def client(set_env, moto_server):
    client = boto3.client(
        "s3",
        endpoint_url="http://localhost:5000",
        config=Config(proxies={"https": "localhost:5000", "http": "localhost:5000"}),
    )
    yield client


@pytest.fixture(scope="package")
@pytest.mark.usefixtures("test_temporary_folder")
def testing_artifacts_folder(test_temporary_folder: str) -> None:
    """This creates a artifacts folder and populates it.

    This folder is used in logging calls within the test suite.

    Parameters
    ----------
    test_temporary_folder: str
        The testing data location fixture.
    """
    artifact_dir = f"{test_temporary_folder}/artifacts"
    os.makedirs(artifact_dir)
    # Write out a list.
    with open(f"{artifact_dir}/list.txt", "w") as f:
        f.write(json.dumps(artifact_list))
    # Write out a numpy array.
    np.save(f"{artifact_dir}/numpy", artifact_np)
    return artifact_dir


_test_data_type = Dict[
    str,
    Union[
        Dataset,
        Dict[str, Parameter],
        Dict[str, Metric],
        Dict[str, Union[str, List[str]]],
    ],
]


@pytest.fixture(scope="package")
@pytest.mark.usefixtures("test_temporary_folder", "testing_artifacts_folder")
def testing_data(test_temporary_folder, testing_artifacts_folder) -> _test_data_type:
    """This creates testing data.

    This testing data is used to test any Thing Store
    implementation.

    Parameters
    ----------
    test_temporary_folder: str
        The testing data location fixture.
    testing_artifacts_folder: str
        The location of the testing artifacts
    """
    return {
        "data": pa.Table.from_pylist(
            [
                {"name": "Bob", "status": "NotAsCool"},
                {"name": "Steve", "status": "TheCoolest"},
            ]
        ),
        "parameters": {"sample": "things", "to": 1, "exemplify": 2.0},
        "metadata": {"example": "metadata", "to": 1, "exemplify": 2.0},
        "metrics": {"thing": 1, "otherthing": 2.0},
        "artifacts": {
            "folder": testing_artifacts_folder,
            "keys": ["list.txt", "numpy.npy"],
        },
    }


@pytest.fixture(scope="package")
@pytest.mark.usefixtures("testing_data")
def integration_test(testing_data: _test_data_type) -> None:
    """Return a testing function.

    This returns a function used to run a simple testing routine
    that logs something and validates that it can be retrieved.
    This is *reused* in other tests.

    This tests a *dataset*, *parameters*, *metadata*, *metrics*,
    and *artifacts*.

    This tests things both locally and remotely.

    Parameters
    ----------
    testing_data: ThingStore
        An instantiated *non-base-class* Thing Store.
    """
    expected_metrics = pd.DataFrame([testing_data["metrics"]]).reset_index(drop=True)
    expected_parameters = pd.DataFrame([testing_data["parameters"]]).reset_index(
        drop=True
    )
    # The metadata is dynamically created at runtime.
    expected_metadata = testing_data["metadata"].copy()
    expected_metadata = pd.DataFrame([expected_metadata]).reset_index(drop=True)

    def test_the_things(
        remote_thing_store: ThingStore, local_thing_store: ThingStore
    ) -> None:
        """Test all the things!

        This tests thing_store API and validates that a thing_store
        construct behaves appropriately. This is called on specific
        instances of a ThingStore construct.

        Parameters
        ----------
        remote_thing_store: ThingStore
            This is a concrete instance of a Thing Store
            implemented in whatever technology. There are some
            expectations on what the construct should be able to
            do, and this construct is poked at to validate.
        local_thing_store: ThingStore
            This is a concrete instance of a Thing Store
            implemented in whatever technology. There are some
            expectations on what the construct should be able to
            do, and this construct is poked at to validate.
        """
        ############################################################
        #                   Standard Workflow                      #
        ############################################################
        # 1. Log!
        file_id = remote_thing_store.log(
            dataset=testing_data["data"],
            parameters=testing_data["parameters"],
            metadata=testing_data["metadata"],
            metrics=testing_data["metrics"],
            artifacts_folder=testing_data["artifacts"]["folder"],
        )
        # Validate that the log occured appropriately.
        assert remote_thing_store._check_file_id(file_identifier=file_id)
        # 1.a: Validate the dataset was recorded appropriately.
        assert (
            remote_thing_store.get_dataset(file_identifier=file_id).to_table()
            == testing_data["data"]
        )
        # 1.b: Validate the parameters were recorded appropriately.
        actual_parameters = pd.DataFrame(
            [remote_thing_store.get_parameters(file_identifier=file_id)]
        ).astype(expected_parameters.dtypes)
        pd.testing.assert_frame_equal(
            left=expected_parameters,
            right=actual_parameters,
            check_dtype=False,
            check_like=True,
        )
        # 1.c: Validate the metadata was recorded appropriately.
        actual_metadata = remote_thing_store.get_metadata(file_identifier=file_id)
        # Here we test to make sure that the standard elements of
        #   metadata are present.
        # Then, we remove them to validate that we record all
        #   the business logic inspired features appropriately.
        _default_metadata = Metadata(**{}).dict()
        assert set(_default_metadata).issubset(set(actual_metadata))
        pd.testing.assert_frame_equal(
            left=expected_metadata,
            #  These two are dynamic.
            right=pd.DataFrame([actual_metadata])
            .astype(expected_metadata.dtypes)
            .drop(columns=list(_default_metadata)),
            check_dtype=False,
            check_like=True,
        )
        # 1.d: Validate the metrics were recorded appropriately.
        actual_metrics = pd.DataFrame(
            [remote_thing_store.get_metrics(file_identifier=file_id)]
        ).astype(expected_metrics.dtypes)
        pd.testing.assert_frame_equal(
            left=expected_metrics,
            right=actual_metrics,
            check_dtype=False,
            check_like=True,
        )
        # 1.e: Validate the artifacts were recorded appropriately.
        _artifacts = remote_thing_store.list_artifacts(file_identifier=file_id)
        assert set(_artifacts) == set(testing_data["artifacts"]["keys"])
        # 2. Browse!
        things = remote_thing_store.browse()
        check_cols = list(testing_data["metadata"].keys())
        # This should be a row with at least the metadata elements and FILE_ID.
        necessary_keys = ["FILE_ID"] + check_cols
        assert isinstance(things, pd.DataFrame)
        assert set(necessary_keys).issubset(things.columns)
        # This frame isn't guaranteed to have the correct types by default.
        left_frame = things[check_cols].astype(
            {"example": "str", "to": "int", "exemplify": "float"}
        )
        right_frame = pd.DataFrame([testing_data["metadata"]])[check_cols]
        right_frame = right_frame.astype(left_frame.dtypes)
        pd.testing.assert_frame_equal(left_frame, right_frame)
        # 3. Load!
        extracted_dataset = remote_thing_store.load(
            file_identifier=file_id, output_format="table"
        )
        assert extracted_dataset == testing_data["data"]
        # 4. Get Artifacts!
        with tempfile.TemporaryDirectory() as t:
            artifact_keys = remote_thing_store.list_artifacts(file_id)
            assert set(artifact_keys) == set(testing_data["artifacts"]["keys"])
            remote_thing_store.get_artifact(
                file_identifier=file_id, artifact_identifier="list.txt", target_path=t
            )
            with open(f"{t}/artifacts/list.txt", "r") as f:
                assert json.loads(f.read()) == artifact_list
            remote_thing_store.get_artifact(
                file_identifier=file_id, artifact_identifier="numpy.npy", target_path=t
            )
            assert np.all(np.load(f"{t}/artifacts/numpy.npy") == artifact_np)
        # 5. Copy this to a local store.
        local_thing_store.copy(file_identifier=file_id, thing_store=remote_thing_store)
        # What does the local_thing_store look like now?
        local_data = local_thing_store.browse()[
            [
                "FILE_ID",
                "FILE_VERSION",
                "example",
                "to",
                "exemplify",
            ]
        ]
        expected_data = pd.DataFrame(
            {
                "FILE_ID": [file_id],
                "FILE_VERSION": ["1"],
                "example": ["metadata"],
                "to": ["1"],
                "exemplify": ["2.0"],
            }
        ).astype(local_data.dtypes)
        pd.testing.assert_frame_equal(
            local_data,
            expected_data,
            check_dtype=False,
            check_exact=False,
            check_names=False,
            check_column_type=False,
        )
        # 6. Validate that we can get the same stuff from the local thing_store!
        _local_artifacts = local_thing_store.list_artifacts(file_identifier=file_id)
        assert set(_local_artifacts) == set(testing_data["artifacts"]["keys"])
        pd.testing.assert_frame_equal(
            left=pd.DataFrame(
                [local_thing_store.get_parameters(file_identifier=file_id)]
            ).astype(expected_parameters.dtypes),
            right=expected_parameters,
            check_dtype=False,
            check_like=True,
        )
        pd.testing.assert_frame_equal(
            left=pd.DataFrame([local_thing_store.get_metadata(file_identifier=file_id)])
            .astype(expected_metadata.dtypes)
            .drop(columns=list(_default_metadata.keys())),
            right=expected_metadata,
            check_dtype=False,
            check_like=True,
        )
        pd.testing.assert_frame_equal(
            left=pd.DataFrame(
                [local_thing_store.get_metrics(file_identifier=file_id)]
            ).astype(expected_metrics.dtypes),
            right=expected_metrics,
            check_dtype=False,
            check_like=True,
        )
        _local_artifacts = local_thing_store.list_artifacts(file_identifier=file_id)
        assert set(_local_artifacts) == set(testing_data["artifacts"]["keys"])
        ############################################################
        #                     Bits and Bobs                        #
        ############################################################
        assert not local_thing_store._test_field_value(
            "NonexistentField", "doesnotmatter"
        )
        assert local_thing_store._test_field_value("FILE_ID", file_id)
        # Check publishing the same file id and same file version increments w/out force.
        # We make many much changes here.
        local_thing_store.log(
            dataset=pd.DataFrame([{"silly": "newstuff"}]),
            parameters={"stupid": 1},
            metadata=local_thing_store.get_metadata(file_id),
            metrics={"nonsense": -20},
        )
        assert local_thing_store.browse().groupby("FILE_ID").size().item() == 2
        # Check publishing the same file id ends up with new file and old file.
        with pytest.raises(NotImplementedError, match="metadata safely"):
            local_thing_store.log(
                metadata=local_thing_store.get_metadata(file_id), force=True
            )
        df_metadata = local_thing_store.browse()
        # Validate that the set of file versions is appropriate.
        assert set(df_metadata.FILE_VERSION.values) == {1, 2}
        # Validate that the 'go get the stuff' commands get the NEW version.
        # load
        newdata = local_thing_store.load(file_id)
        assert newdata.equals(pd.DataFrame([{"silly": "newstuff"}])), newdata
        # list artifacts
        assert local_thing_store.list_artifacts(file_id) == []
        # TODO: Fix this with schema enforcement...
        params = local_thing_store.get_parameters(file_id)
        params["stupid"] = int(float(params["stupid"]))
        assert params == {"stupid": 1}
        # TODO: Fix this with schema enforcement...
        _metadata = local_thing_store.get_metadata(file_id)
        _metadata = {k: str(v) for k, v in _metadata.items()}
        _metadata2 = df_metadata.query("FILE_VERSION==2").to_dict(orient="records")[0]
        _metadata2 = {k: str(v) for k, v in _metadata2.items()}
        assert _metadata == _metadata2
        # metrics
        assert local_thing_store.get_metrics(file_id) == {"nonsense": -20}
        # Aight, now we're going to contract and expand the metadata and log some simple things.
        #####################
        # Contract Metadata #
        #####################
        # In a contraction we're removing fields from the local metadata superset when logging.
        # Obviously this will leave voids in the data.
        contracted_metadata = local_thing_store.get_metadata(file_id)
        # These are the default elements
        _default_elements = Metadata(**{}).dict()
        bits_to_nuke = list(_default_elements.keys())
        for bit in bits_to_nuke:
            contracted_metadata.pop(bit)
        new_id = local_thing_store.log(metadata=contracted_metadata)
        new_metadata = local_thing_store.get_metadata(new_id)
        for bit in bits_to_nuke:
            new_metadata.pop(bit)
        if not new_metadata == contracted_metadata:
            raise Exception(new_metadata, contracted_metadata)
        ###################
        # Expand Metadata #
        ###################
        expanded_metadata = contracted_metadata
        expanded_metadata["totally_new_thing"] = 5
        new_id = local_thing_store.log(metadata=expanded_metadata)
        new_metadata = local_thing_store.get_metadata(new_id)
        # TODO: When schema is enabled, undo this.
        new_metadata["totally_new_thing"] = int(
            float(new_metadata["totally_new_thing"])
        )
        for bit in bits_to_nuke:
            new_metadata.pop(bit)
        assert new_metadata == expanded_metadata

    return test_the_things
