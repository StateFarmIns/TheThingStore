"""Test the ThingStore Node.

A node should be able to sit on top of a given Thing Store,
regardless of *type*, and should be able to represent components
efficiently.

TODO: Transition this to the main testing branch.
Inherit in the pafs / other branches.

"""
# import itertools
# import os
# import pandas as pd
# import pyarrow as pa
# import pyarrow.dataset as ds
# import pytest
# import shutil
# import uuid
# from pyarrow.fs import LocalFileSystem, S3FileSystem
# from managedmetadata.metadata_base import ManagedMetadata
# from managedmetadata.metadata_elements import Metadata
# from managedmetadata.metadata_mlflow import MLFlowManagedMetadata
# from managedmetadata.metadata_node import MMNode
# from managedmetadata.metadata_pa_fs import FileSystemManagedMetadata


# metadata_sets = [
#     (
#         MLFlowManagedMetadata,
#         {
#             "tracking_uri": "tracking",
#             "local_storage_folder": "local_storage",
#         },
#         "MLFlow",
#     ),
#     (
#         FileSystemManagedMetadata,
#         {
#             "metadata_filesystem": LocalFileSystem(),
#             "metadata_file": "metadata.parquet",
#             "metadata_lockfile": "metadata-lockfile.parquet",
#             "output_location": "output",
#         },
#         "LocalFilesystem",
#     ),
#     (
#         FileSystemManagedMetadata,
#         {
#             "metadata_filesystem": S3FileSystem(
#                 endpoint_override="http://localhost:5000", allow_bucket_creation=True
#             ),
#             "metadata_file": "metadata.parquet",
#             "metadata_lockfile": "metadata-lockfile.parquet",
#             "output_location": "output",
#         },
#         "S3Filesystem",
#     ),
# ]


# def _treat_params(params, folder):
#     params = params.copy()
#     newfolder = f"{folder}/{uuid.uuid4().hex}"
#     if (
#         "metadata_filesystem" in params
#         and isinstance(params["metadata_filesystem"], S3FileSystem)
#         and newfolder[1:]
#         and params["metadata_filesystem"].get_file_info(newfolder[1:]).type.name
#         == "NotFound"
#     ):
#         params["metadata_filesystem"].create_dir(newfolder[1:], recursive=True)
#     for param in params:
#         if not param == "metadata_filesystem":
#             params[param] = f"{newfolder}/{params[param]}"
#     return params


# @pytest.fixture(
#     params=[_ for _ in metadata_sets if _[2] != "S3Filesystem"],
#     ids=lambda x: f"Local:{x[2]}",
# )
# @pytest.mark.usefixtures("test_temporary_folder")
# def local_metadata(request, test_temporary_folder):
#     # This creates a local metadata.
#     managed_metadata, params, test_name = request.param
#     new_params = _treat_params(params, test_temporary_folder)
#     yield managed_metadata(**new_params)


# @pytest.fixture(params=metadata_sets, ids=lambda x: f"Remote:{x[2]}")
# @pytest.mark.usefixtures("test_temporary_folder")
# def remote_metadata(request, test_temporary_folder):
#     # This creates a local metadata.
#     managed_metadata, params, test_name = request.param
#     new_params = _treat_params(params, test_temporary_folder)
#     yield managed_metadata(**new_params)


# @pytest.mark.usefixtures("integration_test", "remote_metadata", "local_metadata")
# def test_managed_metadata(integration_test, remote_metadata, local_metadata):
#     """Run the integration test at tests/conftest.py"""
#     integration_test(remote_metadata, local_metadata)


# ####################################################################
# #                          Test Logging                            #
# # ---------------------------------------------------------------- #
# # This section tests the capability of the managed metadata to     #
# #   log and recall components.                                     #
# # This uses the full set of different instances of managed         #
# #   metadata and the full set of combinations of different         #
# #   components.                                                    #
# ####################################################################

# components = {
#     "dataset": pd.DataFrame([{"silly": "example"}]),
#     "parameters": {"i": "have", "simple": "parameters"},
#     "metrics": {"simple": 0, "metrics": 1.2},
#     "artifacts_folder": None,  # This is replaced
#     "metadata": {"simple": "metadata", "for": "testing"},
# }

# _component_combinations = []
# for _ in range(1, len(components)):
#     _component_combinations += list(itertools.combinations(components, _))


# @pytest.fixture(
#     params=_component_combinations,
#     ids=lambda x: "-".join(x),
# )
# def component_combination(request):
#     "This is a fixture which creates combinations of components."
#     return request.param


# def _loader(
#     metadata: ManagedMetadata, node: MMNode, component: str, base_path: str
# ) -> None:
#     """Intelligently attempt to load and test a 'thing'"""
#     file_id = node.file_id
#     if component == "dataset":
#         dataset = metadata.get_dataset(file_identifier=file_id)
#         assert isinstance(dataset, ds.Dataset)
#         assert dataset.schema == pa.schema({"silly": "string"})
#         assert dataset.to_table().to_pylist() == [{"silly": "example"}]
#     elif component == "parameters":
#         parameters = metadata.get_parameters(file_identifier=file_id)
#         assert isinstance(parameters, dict)
#         assert parameters == components["parameters"]
#     elif component == "metrics":
#         metrics = metadata.get_metrics(file_identifier=file_id)
#         assert isinstance(metrics, dict)
#         assert metrics == components["metrics"]
#     elif component == "artifacts_folder":
#         artifacts = metadata.get_artifacts(
#             file_identifier=file_id, target_path=base_path
#         )
#         assert artifacts is None
#         assert set(os.listdir(f"{base_path}/artifacts")) == {"numpy.npy", "list.txt"}
#         # Some day consider testing artifact loading here?
#     elif component == "metadata":
#         _metadata = metadata.get_metadata(file_identifier=file_id)
#         assert isinstance(_metadata, dict)
#         inherent_metadata = {
#             k: v for k, v in _metadata.items() if k in Metadata(**{}).dict()
#         }
#         published_metadata = {
#             k: v for k, v in _metadata.items() if k in components["metadata"]
#         }
#         assert published_metadata == components["metadata"]
#         assert set(inherent_metadata.keys()) == {
#             "FILE_ID",
#             "FILE_VERSION",
#             "DATASET_DATE",
#             "DATASET_VALID",
#             "MM_HAS_EMBEDDING",
#             "MM_HAS_METADATA",
#             "MM_HAS_ARTIFACTS",
#             "MM_HAS_PARAMETERS",
#             "MM_HAS_DATASET",
#             "MM_HAS_FUNCTION",
#             "MM_HAS_METRICS",
#             "MM_IS_NOTIONAL",
#         }
#     else:
#         raise NotImplementedError(component)


# @pytest.mark.usefixtures(
#     "remote_metadata",
#     "component_combination",
#     "testing_artifacts_folder",
#     "test_temporary_folder",
# )
# def test_component_node(
#     remote_metadata,
#     component_combination,
#     testing_artifacts_folder,
#     test_temporary_folder,
# ):
#     """Save component combinations to managed metadata"""
#     # Get the parameters
#     params = {k: v for k, v in components.items() if k in component_combination}
#     if "artifacts_folder" in params:
#         params.update(artifacts_folder=testing_artifacts_folder)
#     # Try to log this job and get a file id.
#     try:
#         file_id = remote_metadata.log(**params)
#     except BaseException as e:
#         raise Exception(f"Broke attempting to log {component_combination}") from e
#     # Now that we've got a file id, let's try to represent it as a Node.
#     # Being able to represent this file id as a node means that I have lazy-load
#     #   access to all the appropriate components.
#     # This node has (potentially):
#     # * A dataset (which itself is a lazy-load data structure)
#     # * Metadata
#     # * Metrics
#     # * Parameters
#     # * Function - Not yet implemented
#     # * Embedding - Not yet implemented
#     # * Artifacts
#     node = MMNode(
#         managedmetadata=remote_metadata,
#         file_id=file_id,
#         file_version=1,
#         persist_dir=None,
#     )
#     ##############
#     # Properties #
#     ##############
#     # Validate properties set appropriately.
#     # File ID, File Version, Metadata, Component Set, Component Map, Artifacts
#     assert str(node) == f"MMNode[{node.file_id}]"
#     assert node.file_id == file_id
#     assert node.file_version == 1
#     assert node.metadata == remote_metadata.get_metadata(file_id)
#     _desired_components = set(_.replace("_folder", "") for _ in component_combination)
#     if not _desired_components == set(node._component_set):
#         raise Exception(
#             "Component Mapping Error",
#             f"Desired Components: {_desired_components}",
#             f"Actual Components: {node._component_set}",
#             f"Node Metadata: {node.metadata}",
#         )
#     # This validates that each component can be called with 'get'
#     test_path = os.path.join(test_temporary_folder, "test_component_logging", file_id)
#     os.makedirs(test_path)
#     _component_errors = {}
#     for component in components:
#         if component in params:  # I *should* be able to get it.
#             try:
#                 _loader(
#                     metadata=remote_metadata,
#                     node=node,
#                     component=component,
#                     base_path=test_path,
#                 )
#             except BaseException as e:
#                 _component_errors.update(component=str(e))
#         else:  # I should *not* be able to get it.
#             assert node.get(component) is None
#     if _component_errors:
#         raise Exception(
#             f"[node.get][component set:({set(components.keys())})]\n", _component_errors
#         )


# def test_component_node_error_bad_fileid():
#     # This cannot use file ids with file://
#     with pytest.raises(NotImplementedError, match="only for relative file ID"):
#         MMNode(
#             managedmetadata="unimportant",
#             file_id="fileid://whatever",
#             file_version=1,
#         )


# def test_component_node_simple(test_temporary_folder):
#     # Here we have a local metadata and we're going to log stuff to it.
#     new_folder = "simple_node_api_test"
#     out_folder = "simple_node_api_test_output"
#     art_folder = "simple_node_api_test_artifact"
#     mm = FileSystemManagedMetadata(
#         metadata_filesystem=LocalFileSystem(),
#         metadata_file=os.path.join(
#             test_temporary_folder, new_folder, "metadata.parquet"
#         ),
#         metadata_lockfile=os.path.join(
#             test_temporary_folder, new_folder, "metadata-lock.parquet"
#         ),
#         output_location=os.path.join(test_temporary_folder, new_folder, "output"),
#     )
#     flid = mm.log(**components)
#     # Now we get each bit out.
#     node = MMNode(
#         managedmetadata=mm,
#         file_id=flid,
#         file_version=1,
#         persist_dir=out_folder,
#     )
#     # I can get a dataset
#     _dataset = node.get_dataset()
#     assert isinstance(_dataset, ds.Dataset)
#     assert _dataset.to_table().to_pandas().equals(components["dataset"])
#     # I can get some parameters
#     _params = node.get_parameters()
#     assert isinstance(_params, dict)
#     assert _params == components["parameters"]
#     # I can get some metadata
#     _metadata = node.get_metadata()
#     assert isinstance(_metadata, dict)
#     # Even more than I put in, in fact!
#     assert {
#         k: v for k, v in _metadata.items() if k in components["metadata"]
#     } == components["metadata"]
#     # I can get some metrics
#     _metrics = node.get_metrics()
#     assert isinstance(_metrics, dict)
#     assert _metrics == components["metrics"]
#     # Can we get embeddings or functions yet? No!
#     with pytest.raises(NotImplementedError, match="not a valid"):
#         node.get_function()
#     with pytest.raises(NotImplementedError, match="not a valid"):
#         node.get_embedding()
#     # But if we do a bit of cheaty cheat we can push past that.
#     node._component_set += ["function", "embedding"]
#     with pytest.raises(NotImplementedError, match="future work"):
#         node.get_function()
#     with pytest.raises(NotImplementedError, match="future work"):
#         node.get_embedding()
#     # What does it look like if I ask for things I don't have?
#     # If I use `get()` then it simply returns None, but if I use `get_X`
#     #   then it blows up!
#     # Did you know that we can log *nothing*?
#     flid = mm.log(**{})
#     node = MMNode(
#         managedmetadata=mm,
#         file_id=flid,
#         file_version=1,
#         persist_dir=out_folder,
#     )
#     with pytest.raises(NotImplementedError, match="not a valid"):
#         node.get_artifacts()
#     with pytest.raises(NotImplementedError, match="not a valid"):
#         node.get_dataset()
#     with pytest.raises(NotImplementedError, match="not a valid"):
#         node.get_metrics()
#     with pytest.raises(NotImplementedError, match="not a valid"):
#         node.get_parameters()
#     # We *do* have metadata, though, because we *always* have metadata.
#     assert node.get_metadata() is not None
#     # Now let's test some artifacts.
#     os.makedirs(os.path.join(test_temporary_folder, art_folder))
#     with open(os.path.join(test_temporary_folder, art_folder, "silly.txt"), "w") as f:
#         f.write("test")
#     flid = mm.log(artifacts_folder=os.path.join(test_temporary_folder, art_folder))
#     # What happens if I don't give the node a persist dir?.
#     node = MMNode(
#         managedmetadata=mm,
#         file_id=flid,
#         file_version=1,
#     )
#     assert node.artifacts == ["silly.txt"]
#     with pytest.raises(Exception, match="with a persist_dir"):
#         node.get_artifacts()
#     # So, let's give it a persist dir.
#     os.makedirs(os.path.join(test_temporary_folder, out_folder))
#     node = MMNode(
#         managedmetadata=mm,
#         file_id=flid,
#         file_version=1,
#         persist_dir=os.path.join(test_temporary_folder, out_folder),
#     )
#     # *Now* what happens?
#     node.get_artifacts()
#     assert os.listdir(node._persist_dir) == ["silly.txt"]
#     # Let's do it again, but targeted.
#     shutil.rmtree(os.path.join(test_temporary_folder, out_folder))
#     os.makedirs(os.path.join(test_temporary_folder, out_folder))
#     assert os.listdir(node._persist_dir) == []
#     node.get_artifacts()
#     assert os.listdir(node._persist_dir) == ["silly.txt"]
#     # Asking for it again doesn't *do* anything.
#     node.get_artifacts("silly.txt")
#     assert os.listdir(node._persist_dir) == ["silly.txt"]
#     # What if we ask for something that doesn't exist?
#     with pytest.raises(FileNotFoundError, match="Node has no"):
#         node.get_artifacts("nope")
#     # Ok, I *know* that get artifacts exists and works.
#     assert node.get("artifacts") == [os.path.join(node._persist_dir, "silly.txt")]
#     # Finally, what happens if I try to add
