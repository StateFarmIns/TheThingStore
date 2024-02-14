"""Represent a FILE_ID as a discrete node."""
import logging
import os
import pyarrow.dataset as ds
from thethingstore.types import FileId
from thethingstore import ThingStore
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class ThingNode:
    """Represent a ThingStore Node.

    This lazy loads components in an *as efficient as possible* way
    to ensure that it can be used in automation effectively and
    quickly.

    Sit this on top of a File ID and this becomes in an-memory
    'ticket' style promise of all the components with constant time
    access to the components logged with the File ID.

    Parameters
    ----------
    thingstore: ThingStore
        An instance of a Thing Store.
    file_id: FileId
        A unique File ID within that Thing Store.
    file_version: int
        The desired file version for the File ID.
    persist_dir: Optional[str] = None
        A local directory in which to host files.
        This is required for working with artifacts.
    """

    def __init__(
        self,
        thingstore: ThingStore,
        file_id: FileId,
        file_version: int,
        persist_dir: Optional[str] = None,
    ):
        self._log_header = f"ThingNode({file_id}):"
        ############################################################
        #                     Build the Node                       #
        ############################################################
        # This section validates that it can access the metadata,
        #   that it can list artifacts, that it can get the dataset,
        #   and that every single *.get_X is appropriately accounted
        #   for.
        if file_id.startswith("fileid"):
            raise NotImplementedError(
                "This is only for relative file ID.",
            )
        self.file_id = file_id
        self.file_version = file_version
        # This needs to handle schema appropriately!
        self._ts = thingstore
        # TODO: implement file_version query in get_metadata.
        self.metadata = self._ts.get_metadata(file_id)
        component_set = []
        for attr_name, attr_value in self.metadata.items():
            if attr_name.upper().startswith("MM_HAS_") and attr_value:
                component_set.append(attr_name.replace("MM_HAS_", "").lower())
        self._component_set = component_set
        self.artifacts = thingstore.list_artifacts(file_id)
        self._persist_dir = persist_dir
        self._component_map = _build_component_map(
            self, component_set=self._component_set
        )

    def get(self, attr: str) -> Optional[Any]:
        """Retrieve a node component.

        This will return None if the node does not have the component.

        Parameters
        ----------
        attr: str
            The name of the component, i.e. 'dataset'

        Returns
        -------
        component: Optional[Any]
            The desired component, or None.
        """
        if not attr.lower() in self._component_set:
            return None
        return self._component_map[attr]()


    def get_artifacts(self, artifact: Optional[str] = None) -> Union[str, List[str]]:
        """Retrieve node artifacts.

        This will raise an Exception if the artifact(s) don't exist.
        If no specific artifact is passed this will get *all* artifacts.
        This requires that the node is instantiated with a persist_dir.

        Parameters
        ----------
        artifact: Optional[str] = None
            The name of the artifact, i.e. 'whatever'

        Returns
        -------
        artifact_path: str
            The path to the locally saved artifact(s).
        """
        if not self.artifacts:  # This will always be a list, just potentially empty.
            raise NotImplementedError(
                self._log_header + "Artifacts not a valid option."
            )
        if self._persist_dir is None:
            raise Exception("To use artifacts init with a persist_dir.")
        os.makedirs(self._persist_dir, exist_ok=True)
        cur_artifacts = os.listdir(self._persist_dir)
        if artifact is not None and artifact in cur_artifacts:
            logger.warning(f"Artifact ({artifact}) already exists. No action.")
            return os.path.join(self._persist_dir, artifact)
        elif artifact is not None and artifact not in self.artifacts:
            raise FileNotFoundError(f"Node has no {artifact} artifact.")
        elif artifact is not None:  # Artifact not local.
            self._ts.get_artifact(
                file_identifier=self.file_id,
                artifact_identifier=artifact,
                target_path=os.path.join(self._persist_dir, artifact),
            )
            return os.path.join(self._persist_dir, artifact)
        else:
            paths = []
            for artifact in self.artifacts:
                _artifact = self.get_artifacts(artifact)
                # MyPy thinks that this could be a list.
                # MyPy is wrong.
                assert isinstance(_artifact, str)  # nosec
                paths.append(_artifact)
            return paths


    def get_dataset(self) -> ds.Dataset:
        """Retrieve the node dataset.

        This will raise an Exception if the node has no dataset.

        Returns
        -------
        dataset: ds.Dataset
            The dataset of the node.
        """
        if "dataset" not in self._component_set:
            raise NotImplementedError(self._log_header + "Dataset not a valid option.")
        return self._ts.get_dataset(self.file_id)


    def get_metadata(self) -> Mapping[str, Any]:
        """Retrieve the node metadata.

        This *always* returns information; the Managed Metadata
        specifies a default set of metadata.

        Returns
        -------
        metadata: Dict[str, Any]
            The metadata of the node.
        """
        return self.metadata

    def get_metrics(self) -> Mapping[str, Union[str, int, float]]:
        """Retrieve the node metrics.

        This will raise an Exception if the node has no metrics.

        Returns
        -------
        metrics: Dict[str, Union[str, int, float]]
            The metrics of the node.
        """
        if "metrics" not in self._component_set:
            raise NotImplementedError(self._log_header + "Metrics not a valid option.")
        return self._ts.get_metrics(self.file_id)

    def get_parameters(self) -> Mapping[str, Any]:
        """Retrieve the node parameters.

        This will raise an Exception if the node has no parameters.

        Returns
        -------
        parameters: Dict[str, Any]
            The parameters of the node.
        """
        if "parameters" not in self._component_set:
            raise NotImplementedError(
                self._log_header + "Parameters not a valid option."
            )
        return self._ts.get_parameters(self.file_id)

    def get_embedding(self) -> None:
        """Retrieve the node embedded representation.

        This will raise an Exception if the node has no embedding.

        Returns
        -------
        embedding: Any
            The embedding of the node.
        """
        if "embedding" not in self._component_set:
            raise NotImplementedError(
                self._log_header + "Embedding not a valid option."
            )
        raise NotImplementedError("Embeddings are future work.")

    def get_function(self) -> Callable:
        """Retrieve the node function.

        This will raise an Exception if the node has no function.

        Returns
        -------
        function: Callable
            The function of the node.
        """
        if "function" not in self._component_set:
            raise NotImplementedError(self._log_header + "Function not a valid option.")
        raise NotImplementedError("Functions are future work.")

    def __repr__(self) -> str:
        # TODO: Represent this as a full fileid?
        return f"ThingNode[{self.file_id}]"

def _get_graph(node: ThingNode) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Make data structures for nodes and edges for a ThingNode graph."""
    raise NotImplementedError("SAD")

def _build_component_map(
    node: ThingNode, component_set: list[str]
) -> Mapping[str, Callable]:
    "Build and return component mapping."
    component_map = {}
    for _attr in component_set:
        if _attr == "dataset":
            component_map[_attr] = node.get_dataset
        elif _attr == "metadata":
            component_map[_attr] = node.get_metadata
        elif _attr == "metrics":
            component_map[_attr] = node.get_metrics
        elif _attr == "parameters":
            component_map[_attr] = node.get_parameters
        elif _attr == "function":
            component_map[_attr] = node.get_function  # pragma: no cover
        elif _attr == "embedding":
            component_map[_attr] = node.get_embedding  # pragma: no cover
        elif _attr == "artifacts":
            component_map[_attr] = node.get_artifacts
    return component_map