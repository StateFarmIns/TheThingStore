"""Test edge cases."""
import os
import pytest
from thethingstore.thing_store_mlflow import MLFlowThingStore


@pytest.mark.usefixtures("test_temporary_folder")
def test_no_uri(test_temporary_folder):
    test_folder = "mlflow_no_uri_test"
    tgt_path = os.path.join(test_temporary_folder, test_folder)
    os.makedirs(tgt_path)
    ts = MLFlowThingStore(
        local_storage_folder=os.path.join(test_temporary_folder, test_folder)
    )
    assert ts._mlflow_client.tracking_uri == f"file://{tgt_path}"


####################################################################
#                 Environment Mismatch Testing                     #
# ---------------------------------------------------------------- #
# 20230120 - Tests locally have 100% coverage and in gitlab have   #
#   ~90% coverage for the mlflow implementation. This section      #
#   covers the specific elements not covered 'upstream' in gitlab. #
####################################################################


@pytest.mark.usefixtures("test_temporary_folder")
def test_list_artifacts_with_no_artifacts(test_temporary_folder):
    test_folder = "mlflow_list_artifacts_with_no_artifacts"
    tgt_path = os.path.join(test_temporary_folder, test_folder)
    os.makedirs(tgt_path)
    ts = MLFlowThingStore(
        local_storage_folder=os.path.join(test_temporary_folder, test_folder)
    )
    ts.log(metadata={"FILE_ID": "TestingThingy"})
    assert ts.list_artifacts("TestingThingy") == []
