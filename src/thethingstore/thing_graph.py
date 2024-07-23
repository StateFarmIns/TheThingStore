"""Utilities to build Graphs."""
import tempfile
import pandas as pd
from thethingstore.thing_node import ThingNode, Nodes, Edges
from thethingstore.types import DataLayer, FileId
from typing import List, Optional, Tuple, Mapping, Union, Any


class ThingGraph:
    def __init__(self, thing_node: ThingNode, depth: int = 1):
        self._data_layer = thing_node._data_layer
        nodes, edges = get_graph(thing_node, depth=depth)
        self.nodes = nodes
        self.edges = edges

    def plot(self, depth: int = 1, filename: str = "graph.html") -> None:
        plot_graph(self, depth=depth).show(filename)


def _dec_depth(depth: Optional[int] = None) -> Optional[int]:
    """Decrement depth."""
    if depth is not None:  # We don't want to mindlessly continue.
        depth -= 1  # So, bring it down a bit.
    return depth


def _file_test(key: str) -> bool:
    """Determine if key is fileid."""
    # This can be replaced with an arbitrary test.
    if key.startswith("fileid") or key.endswith("fileid"):
        return True
    else:
        return False


def get_back_edges(
    data_layer: DataLayer, fileid: FileId, depth: Optional[int] = 1
) -> List[Tuple[str, str]]:
    """Get Thing Dependencies from Parameters.

    This uses an assumption that you are using parameters
    of your work to refer to FILEID and then passing *specific*
    FILEID when you go to use it.
    It assumes that you do that like this:

    ```python
    params = {
        'my_dataset_fileid': '2024_whatever_data_uniqueid'
    }
    ```

    Parameters
    ----------
    data_layer: DataLayer
        This is a ThingStore-API compliant data layer.
    fileid: FileID
        This is a descriptive pointer for a file in the data layer.
    depth: int = 1
        Only go this far back. 'As far as it goes' if None.

    Returns
    -------
    edges: List[Tuple[str, str]]
        This is a list of edges
    """
    depth = _dec_depth(depth)
    # p at this point is the set of parameters representing upstream files.
    p = {k: v for k, v in data_layer.get_parameters(fileid).items() if _file_test(k)}
    if not p:  # No parameters exist.
        # This is 'terminal', meaning that none of my parameters are fileid
        return []
    # Now, extract these as edges.
    # Note this is DIRECTIONAL and is [(FROM, TO)]
    edges: List[Tuple[str, str]] = [(v, fileid) for v in p.values()]
    if depth is not None and depth > 0:  # Enhance...
        # I have a list of fileid, to which I need to apply a function.
        # Each of these returns a list of string, thus, I need to collapse.
        # This is a single loop which walks over all values in a loop over lists.
        _p = []
        for edge in edges:
            (upstream_fileid, _) = edge
            _p.extend(  # Jam it in
                # Here's a new list of edges.
                get_back_edges(
                    data_layer=data_layer, fileid=upstream_fileid, depth=depth
                )
            )
        edges.extend(_p)
    return edges


def get_forward_edges(
    data_layer: DataLayer, fileid: FileId, depth: Optional[int] = 1
) -> List[str]:
    """Get Thing Dependents ... somehow?

    This uses an assumption, yet to be determined, which will
    allow for implicitly determining which, amongst a set of
    files, were output by this specific fileid.

    Parameters
    ----------
    data_layer: DataLayer
        This is a ThingStore-API compliant data layer.
    fileid: FileID
        This is a descriptive pointer for a file in the data layer.
    depth: int = 1
        Only go this far forwards. 'As far as it goes' if None.

    Returns
    -------
    edges: List[Tuple[str, str]]
        This is a list of edges
    """
    depth = _dec_depth(depth)
    raise NotImplementedError


def get_edges(
    data_layer: DataLayer, fileid: FileId, depth: int = 1
) -> List[Tuple[str, str]]:
    """Get the edges attached to this particular file.

    Parameters
    ----------
    data_layer: DataLayer
        This is a ThingStore-API compliant data layer.
    fileid: FileID
        This is a descriptive pointer for a file in the data layer.
    depth: int = 1
        Only go this far forwards/backwards. 'As far as it goes' if None.

    Returns
    -------
    edges: List[Tuple[str, str]]
        This is a list of edges
    """
    return get_back_edges(
        data_layer=data_layer,
        fileid=fileid,
        depth=depth,
    )
    # + get_forward_edges(
    #     data_layer=data_layer,
    #     fileid=fileid,
    #     depth=depth,
    # )


def append_graph(
    data_layer: DataLayer, fileid: FileId, nodes: Nodes, edges: Edges
) -> None:
    # Update a file with a graph artifact.
    nodes_edges = {"nodes": nodes, "edges": edges}
    nodes_df = pd.DataFrame(data={"nodes": nodes_edges["nodes"]})
    edges_df = pd.DataFrame(data={"edges": nodes_edges["edges"]})
    with tempfile.TemporaryDirectory() as t:
        # Write new graph artifacts
        nodes_df.to_parquet(f"{t}/graph/nodes.parquet")
        edges_df.to_parquet(f"{t}/graph/edges.parquet")
        # Log new version
        data_layer.update(artifacts_folder=t)


def get_graph(node: ThingNode, depth: int = 1) -> Tuple[Nodes, Edges]:
    """Return nodes and edges data structures for a ThingNode graph.

    If artifacts for the nodes and edges already exist, grab those.
    If not, build the graph implicitly from the workflow parameters.

    Graph representation structure:

    ```python
    nodes = ["node1", "node2", ..., "nodeX"],
    edges = [("source1", "target1"), ("source2", "target2"), ..., ("sourceX", "targetX")]
    ```

    Parameters
    ----------
    node: ThingNode
        The instantiated ThingNode object.

    Returns
    -------
    (nodes, edges): Tuple[Nodes, Edges]
    """
    # If there are already graph artifacts, use them.
    artifacts = node._ts.list_artifacts(node.file_id)
    if ("graph/nodes.parquet" in artifacts) and ("graph/edges.parquet" in artifacts):
        node.get_artifacts(artifact="graph")
        # Get nodes data
        node_graph_path = f"{node._persist_dir}/artifacts/graph/nodes.parquet"
        nodes_df = pd.read_parquet(node_graph_path)
        nodes_dict = nodes_df.to_dict(orient="list")
        # Get edges data
        edges_graph_path = f"{node._persist_dir}/artifacts/graph/edges.parquet"
        edges_df = pd.read_parquet(edges_graph_path)
        edges_dict = edges_df.to_dict(orient="list")
        # Return graph
        return nodes_dict["nodes"], edges_dict["edges"]
    else:
        return build_graph(node, depth=depth)


def build_graph(node: ThingNode, depth: int = 1) -> Tuple[Nodes, Edges]:
    """Make data structures for nodes and edges for a ThingNode graph."""
    edges = get_edges(data_layer=node._data_layer, fileid=node.file_id, depth=depth)
    nodes = []
    for edge in edges:
        for n in edge:
            nodes.append(n)
    return list(set(nodes)), edges


####################################################################
#                            Plotting                              #
# ---------------------------------------------------------------- #
# Everything below this point does not need to be tested.          #
####################################################################


def get_node_description(
    data_layer: DataLayer, node: FileId, metadata_tags: Optional[List[str]] = None
) -> Mapping[str, Union[str, int, float]]:
    _metadata = data_layer.get_metadata(node)
    if metadata_tags is None:
        metadata_tags = list(_metadata.keys())
    metadata_str = ""
    for k in metadata_tags:
        metadata_str += f"\n\t* {k}: {_metadata[k]}"
    return {
        "label": node,
        "shape": "ellipse",
        "title": f"""Thing[{node}]

Metadata
--------
{metadata_str}
        """,
    }


def plot_graph(thing_graph: ThingGraph, depth: int = 1) -> Any:
    """Visually represent a ThingGraph.

    Parameters
    ----------
    thing_graph: ThingGraph
        This represents a Thing.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise RuntimeError("Run `pip install pyvis`")
    except ModuleNotFoundError:
        raise RuntimeError("Run `pip install pyvis`")
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
    for node in thing_graph.nodes:
        g.add_node(node, **get_node_description(thing_graph._data_layer, node))
    # Start adding edges to it.
    for edge in thing_graph.edges:
        g.add_edge(*edge)
    return g
