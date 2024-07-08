"""Represent a FILE_ID as a discrete node.

This contains the ThingNode class and some plotting functionality.
"""
import logging
import os
import pyarrow.dataset as ds
from thethingstore.types import FileId
from thethingstore import ThingStore
from typing import Any, Callable, List, Mapping, Optional, Tuple, Union

DataLayer = ThingStore

logger = logging.getLogger(__name__)

Node = str
UpstreamNode = Node
DownstreamNode = Node
DirectedEdge = Tuple[UpstreamNode, DownstreamNode]
Nodes = List[Node]
Edges = List[DirectedEdge]


class ThingNode:  # pragma: no cover
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

    def __init__(  # pragma: no cover
        self,
        thingstore: ThingStore,
        file_id: FileId,
        file_version: int = 1,
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
        # TODO: Patternise the set/get by looping over componens.
        if file_id.startswith("fileid"):
            raise NotImplementedError(
                "This is only for relative ('local') file ID.",
            )
        self._graph = None
        self.file_id = file_id
        self.file_version = file_version
        # This needs to handle schema appropriately!
        self._ts = thingstore
        self._data_layer = thingstore
        # TODO: implement file_version query in get_metadata.
        self.metadata = self._ts.get_metadata(file_id)
        component_set = []
        for attr_name, attr_value in self.metadata.items():
            # TS_HAS could be replaced with a regular expression.
            if attr_name.upper().startswith("TS_HAS_") and attr_value:
                component_set.append(attr_name.replace("TS_HAS_", "").lower())
        self._component_set = component_set
        self.artifacts = thingstore.list_artifacts(file_id)
        self._persist_dir = persist_dir
        self._component_map = _build_component_map(
            self, component_set=self._component_set
        )

    def get(self, attr: str) -> Optional[Any]:  # pragma: no cover
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

    def get_artifacts(
        self, artifact: Optional[str] = None
    ) -> Union[str, List[str]]:  # pragma: no cover
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
                target_path=self._persist_dir,
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

    def get_dataset(self) -> ds.Dataset:  # pragma: no cover
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

    def get_metadata(self) -> Mapping[str, Any]:  # pragma: no cover
        """Retrieve the node metadata.

        This *always* returns information; the Managed Metadata
        specifies a default set of metadata.

        Returns
        -------
        metadata: Dict[str, Any]
            The metadata of the node.
        """
        return self.metadata

    def get_metrics(self) -> Mapping[str, Union[str, int, float]]:  # pragma: no cover
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

    def get_parameters(self) -> Mapping[str, Any]:  # pragma: no cover
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

    def get_embedding(self) -> None:  # pragma: no cover
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
        return self._ts.get_embedding(self.file_id)

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
        # Mypy thinks this could return None
        #   mypy forgot to look at the line above
        return self._ts.get_function(self.file_id)  # type: ignore

    def __repr__(self) -> str:
        # TODO: Represent this as a full fileid?
        return f"ThingNode[{self.file_id}]"

    def plot(self) -> None:
        plot_node(self).show(f"{self}.html")


def _build_component_map(  # pragma: no cover
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


####################################################################
#                            Plotting                              #
# ---------------------------------------------------------------- #
# Everything below this point does not need to be tested.          #
####################################################################


def _get_component_nodes_edges(
    thing_node: ThingNode,
) -> Tuple[Nodes, Edges]:  # pragma: no cover
    if "function" in thing_node._component_set:
        return _get_functional_component_nodes_edges(thing_node)
    else:
        return _get_nonfunctional_component_nodes_edges(thing_node)


def _get_functional_component_nodes_edges(
    thing_node: ThingNode,
) -> Tuple[Nodes, Edges]:  # pragma: no cover
    """Build functional nodes and edges."""
    nodes = ["source", "sink"]
    edges = []
    if "parameters" in thing_node._component_set:
        nodes.append("parameters")
        edges.append(("source", "parameters"))
        nodes.append("function")
        edges.append(("parameters", "function"))
        edges.append(("function", "sink"))
    else:
        nodes.append("function")
        edges.append(("source", "function"))
        edges.append(("function", "sink"))
    if "metadata" in thing_node._component_set:
        nodes.append("metadata")
        edges.append(("function", "metadata"))
        edges.append(("metadata", "sink"))
    if "metrics" in thing_node._component_set:
        nodes.append("metrics")
        edges.append(("function", "metrics"))
        edges.append(("metrics", "sink"))
    if "dataset" in thing_node._component_set:
        nodes.append("dataset")
        edges.append(("function", "dataset"))
        edges.append(("dataset", "sink"))
    if "embedding" in thing_node._component_set:
        nodes.append("embedding")
        edges.append(("function", "embedding"))
        edges.append(("embedding", "sink"))
    if "artifacts" in thing_node._component_set:
        nodes.append("artifacts")
        edges.append(("function", "artifacts"))
        edges.append(("artifacts", "sink"))
    return nodes, edges


def _get_nonfunctional_component_nodes_edges(  # pragma: no cover
    thing_node: ThingNode,
) -> Tuple[Nodes, Edges]:
    """Build non-functional nodes and edges."""
    nodes = ["source", "sink"]
    edges = []
    if "parameters" in thing_node._component_set:
        nodes.append("parameters")
        edges.append(("source", "parameters"))
        edges.append(("parameters", "sink"))
    if "metadata" in thing_node._component_set:
        nodes.append("metadata")
        edges.append(("source", "metadata"))
        edges.append(("metadata", "sink"))
    if "metrics" in thing_node._component_set:
        nodes.append("metrics")
        edges.append(("source", "metrics"))
        edges.append(("metrics", "sink"))
    if "dataset" in thing_node._component_set:
        nodes.append("dataset")
        edges.append(("source", "dataset"))
        edges.append(("dataset", "sink"))
    if "embedding" in thing_node._component_set:
        nodes.append("embedding")
        edges.append(("source", "embedding"))
        edges.append(("embedding", "sink"))
    if "artifacts" in thing_node._component_set:
        nodes.append("artifacts")
        edges.append(("source", "artifacts"))
        edges.append(("artifacts", "sink"))
    return nodes, edges


def get_component_description(  # noqa: C901
    thing_node: ThingNode, component: str
) -> Mapping[str, Union[str, int, float]]:  # pragma: no cover
    if component == "source":
        return {
            "label": "Source",
            "mass": 20,
            "shape": "database",
            "title": """The Source Node
This simply represents the point at which information flow comes
into a Thing.""",
        }
    elif component == "sink":
        return {
            "label": "Sink",
            "mass": 20,
            "shape": "triangle",
            "title": """The Sink Node
This simply represents the point at which information flow flows
out of a Thing.""",
        }
    elif component == "metadata":
        m_str = ""
        for k, v in thing_node.get_metadata().items():
            m_str += f"\n\t* {k}: {v}"
        return {
            "label": "Metadata",
            "shape": "ellipse",
            "title": f"""Metadata Labels\n{m_str}""",
        }
    elif component == "metrics":
        m_str = ""
        for k, v in thing_node.get_metrics().items():
            m_str += f"\n\t* {k}: {v}"
        return {
            "label": "Metrics",
            "shape": "ellipse",
            "title": f"""Measure Values\n{m_str}""",
        }
    elif component == "parameters":
        m_str = ""
        for k, v in thing_node.get_parameters().items():
            m_str += f"\n\t* {k}: {v}"
        return {
            "label": "Parameters",
            "shape": "ellipse",
            "title": f"""Parameter Values\n{m_str}""",
        }
    elif component == "dataset":
        d = thing_node.get_dataset()
        n = d.count_rows()
        m = len(d.schema)
        schema_dict = {k: str(v) for k, v in zip(d.schema.names, d.schema.types)}
        m_str = ""
        for k, v in schema_dict.items():
            m_str += f"\n\t* {k}: {v}"
        return {
            "label": "Dataset",
            "shape": "ellipse",
            "title": f"""Dataset of size [{m}x{n}]\n{m_str}""",
        }
    elif component == "artifacts":
        return {
            "label": "Dataset",
            "shape": "ellipse",
            "title": f"""Artifacts\n{thing_node.get_artifacts()}""",
        }
    elif component == "embedding":
        raise NotImplementedError("Extract embed size")
        m = 0
        n = 0
        return {
            "label": "Embedding",
            "shape": "ellipse",
            "title": f"""Embedded Representation of size [{m}x{n}]""",
        }
    elif component == "function":
        raise NotImplementedError("Extract function and description")
        return {"label": "Function", "shape": "ellipse", "title": """Function"""}
    else:
        raise NotImplementedError(f"{component} is not implemented")


def plot_node(thing_node: ThingNode) -> Any:  # pragma: no cover
    """Visually represent a ThingNode.

    Parameters
    ----------
    thing_node: ThingNode
        This represents a Thing.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise RuntimeError("Run `pip install pyvis`")
    # Get the nodes and edges.
    nodes, edges = _get_component_nodes_edges(thing_node)
    # Make the graph
    g = Network(
        notebook=True,
        directed=True,
        # neighborhood_highlight=True
        # select_menu = True,
        # layout='hierarchical',  # Couldn't get this to work right in studio
        # bgcolor="#222222",
        # font_color="white",
    )

    # Start adding nodes to it.
    for node in nodes:
        g.add_node(node, **get_component_description(thing_node, node))
    # Start adding edges to it.
    for edge in edges:
        g.add_edge(*edge)
    return g
