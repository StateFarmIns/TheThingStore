"""Holds step functions and utilities.

Miscellaneous Data Elements
---------------------------

* rng - Happy little random number generator.
* feature_map - List of 1000 features with associated 'scores'.
* functions - Contains required information to simulate next-step probability.

Project Information
-------------------
* random_project_generator - Create a bunch of projects.

Step Functions
--------------

* get_step_files - Get the latest project state files.
* step - Do next step simulation for all projects.
* batch_step - Derive next step for all unfinished projects.
* single_step - Execute next step for a single project.
* DrawData - Go get some data.
* ModelReady - Clean it up.
* Model - Run it through a model.
* Aggregate - Aggregate the results.
* Review - Review everything.
* Publish - Send it!

Utilities
---------

* get_p_table - Get the table detailing next step probabilities for all states.
* get_most_recent - Fetch most recent thing of type X.
* get_random_features - generate some random features.
* convenience_table - Used in plotting.
* vis_stepwise_thing_dist - Plotting!

"""
import copy
import ibis
import ibis.selectors as s
import numpy as np
import pandas as pd
from ibis import _
from numpy.random import default_rng
from numpy.random._generator import Generator
from plotnine import ggplot, aes, geom_col, theme
from thethingstore import ThingStore as DataLayer
from thethingstore.types import FileId
from typing import List, Mapping, Optional

rng = default_rng(13287123)

feature_map = {
    k: v
    for k, v in zip(
        [f"feat_{i}" for i in range(100)], list(rng.normal(0.6, 0.3, 100).clip(0, 1))
    )
}

functions = {
    "DrawData": {
        "thing": {
            "metadata": {
                "PROJECT": "fillmein",
                "JURISDICTION": "fillmein",
                "THINGTYPE": "DrawData",
            },
            "parameters": {"sql": """SELECT * FROM TBL WHERE JURISDICTION LIKE {}"""},
        },
        "description": "Accept SQL, run it against a db, and store the output in the data layer",
        "next_node": {
            "ModelReady": 0.65,  # Success, the data has been pulled and has gone through review (hopefully)
            "DrawData": 0.35,  # This is a failure. Your data is flawed, back to the SQL!
        },
        "graph_properties": {
            "color": "#43DB18",
            "shape": "database",
            # borderWidth
            # borderWidthSelected
            # brokenImage
            # group
            # hidden
            # image
            # labelHighlightBold
            # 'level': 0,  # Only with hierarchical layout
            # mass
            # physics
            # shape
            # size
            # title
            # value
            # x
            # y
        },
    },
    "ModelReady": {
        "description": "Take a dataset, make transformations, and bring it through review.",
        "next_node": {
            "Model": 0.8,  # Success, the transformation is well documented and purposeful and tested (hopefully)
            "ModelReady": 0.15,  # Failure, the review is poor, or the tests fail.
            "DrawData": 0.05,  # Failure, a critical flaw in data is discovered.
        },
        "graph_properties": {
            "color": "#DBD94B",
            "shape": "ellipse",
            # borderWidth
            # borderWidthSelected
            # brokenImage
            # group
            # hidden
            # image
            # labelHighlightBold
            # 'level': 1,  # Only with hierarchical layout
            # mass
            # physics
            # shape
            # size
            # title
            # value
            # x
            # y
        },
    },
    "Model": {
        "description": "Build Ferrari or Jalopy on top of ModelReady and bring through review.",
        "next_node": {
            "Model": 0.6,  # I wouldn't call it failure, but it's not yet ready for the world.
            "ModelReady": 0.3,  # Failure, additional transformation needed to further explore modeling space.
            "Aggregate": 0.09,  # Success!
            "DrawData": 0.01,  # Data failure!
        },
        "graph_properties": {
            "color": "#AF18DB",
            "shape": "ellipse",
            # borderWidth
            # borderWidthSelected
            # brokenImage
            # group
            # hidden
            # image
            # labelHighlightBold
            # 'level': 2,  # Only with hierarchical layout
            # mass
            # physics
            # shape
            # size
            # title
            # value
            # x
            # y
        },
    },
    "Aggregate": {
        "description": "Combine modeled group outputs; purely functional.",
        "next_node": {
            "Model": 0.1,  # Need to tweak modeling parameters.
            "ModelReady": 0.1,  # Need to tweak data.
            "Review": 0.8,  # Things look good!
        },
        "graph_properties": {
            "color": "#54385C",
            "shape": "ellipse",
            # borderWidth
            # borderWidthSelected
            # brokenImage
            # group
            # hidden
            # image
            # labelHighlightBold
            # 'level': 3,  # Only with hierarchical layout
            # mass
            # physics
            # shape
            # size
            # title
            # value
            # x
            # y
        },
    },
    "Review": {
        "description": "Validate combined output from process; expert judgement.",
        "next_node": {
            "Model": 0.3 / 2,  # Something seems amiss with the model, go take a look.
            "ModelReady": 0.3
            / 2,  # Something seems amiss with the data, go take a look.
            "Publish": 0.7,
        },
        "graph_properties": {
            "color": "#405C38",
            "shape": "dot",
            # borderWidth
            # borderWidthSelected
            # brokenImage
            # group
            # hidden
            # image
            # labelHighlightBold
            # 'level': 4,  # Only with hierarchical layout
            # mass
            # physics
            # shape
            # size
            # title
            # value
            # x
            # y
        },
    },
    "Publish": {
        "description": "Send it!",
        "next_node": None,
        "graph_properties": {
            "color": "#4CDBC2",
            "shape": "dot",
            # borderWidth
            # borderWidthSelected
            # brokenImage
            # group
            # hidden
            # image
            # labelHighlightBold
            # 'level': 5,  # Only with hierarchical layout
            # mass
            # physics
            # shape
            # size
            # title
            # value
            # x
            # y
        },
    },
}


def random_project_generator(
    data_layer: DataLayer, n_projects: int = 100, seed: int = 1782364
) -> List[FileId]:
    """This generates example start points.

    The `functions` dictionary contains a set of functions used in an
    example modeling routine.

    Each contains a Thing with potentially delayed and / or functional
    components, a description of the 'work' done in this step, and
    the set of potential next nodes along with the probability by node.
    This prior distribution is a SWAG. This is a VERY SIMPLE PROJECT.

    This takes a data layer, assumed to be managing this and similar projects,
    and begins to log these values to that data layer.
    """

    # Step 1: Initialize the projects!
    # Turn on the good old random number generator
    rng = default_rng(seed)
    # This is going to make a number of default projects by inserting data draws.
    logged_draw_data_actions = []
    print("Generating projects")
    for i in range(n_projects):
        if i % 10 == 0:
            print(f"Generating project: {i+1}")
        # Go ahead and get a copy of the drawdata thing
        _step_func = copy.deepcopy(functions["DrawData"]["thing"])
        jurisdiction_value = rng.integers(0, 10, 1)[0]
        _step_func["metadata"].update(
            PROJECT=rng.integers(10000, 100000000),
            JURISDICTION=jurisdiction_value,
            THINGTYPE="DrawData",
        )
        _step_func["parameters"]["sql"] = _step_func["parameters"]["sql"].format(
            jurisdiction_value
        )
        logged_draw_data_actions.append(data_layer.log(**_step_func))

    return logged_draw_data_actions


def get_step_files(
    data_layer: DataLayer,
    latest=False,
) -> List[FileId]:
    """"""
    # Collect the metadata table (TODO - prefilter)
    t = ibis.memtable(data_layer.browse())
    step_files = t.filter(_.PROJECT == -999)
    if not latest:
        return step_files.FILE_ID.execute().to_list()
    else:
        return step_files.FILE_ID.argmax(step_files.DATASET_DATE).execute()
    return


def step(
    data_layer: DataLayer,
) -> FileId:
    # Collect the metadata table (todo - prefiltering)
    t = ibis.memtable(data_layer.browse())
    # Simulate the next step action (this simply returns 'What will I do next, here'.)
    current_step_files = data_layer.load(
        get_step_files(data_layer, latest=True)
    ).FILE_IDS.to_list()

    next_step = batch_step(data_layer=data_layer, current_state=current_step_files)

    # Execute the randomly selected steps.
    steps = next_step.execute().apply(
        lambda x: single_step(data_layer, x.FILE_ID, x.next_step, t, rng), axis=1
    )

    # Summarize that activity
    project_state_fileid = data_layer.log(
        dataset=pd.DataFrame({"FILE_IDS": steps}),
        metadata={"PROJECT": -999, "THINGTYPE": "project_steps"},
    )
    print(f"""Project State Logged. Summary File: {project_state_fileid}""")
    return project_state_fileid


def batch_step(
    data_layer: DataLayer,
    current_state: List[FileId],
) -> List[FileId]:
    """Pick up all unfinished projects and derive the next step."""
    # These are the probabilities of the next step.
    p_table = ibis.memtable(get_p_table())
    # Downselect to unfinished projects, get the latest step,
    # and attach the next step sets of probabilities explicitly.
    # Use those next step probabilities to conduct a random draw.
    t = (
        ibis.memtable([data_layer.get_metadata(_) for _ in current_state])
        .filter(
            # EXCLUDE EXPERIMENT METADATA
            _.PROJECT
            != -999
        )
        .group_by("PROJECT")
        .mutate(
            # EXCLUDE COMPLETED PROJECTS
            complete=(_.THINGTYPE == "Publish").any(),
            # RETAIN ONLY LATEST
            latest=(_.DATASET_DATE == _.DATASET_DATE.max()),
        )
        .filter(
            # EXCLUDE COMPLETED PROJECTS
            _.complete is not True,
            # RETAIN ONLY LATEST
            _.latest is True,
        )
        .drop(
            # Get rid of the extra info we tucked in
            "complete",
            "latest",
        )
        .mutate(
            # Draw a single number from a uniform random distribution per project
            p=ibis.random()
        )
        .join(
            # Bring in the 'next step a and associated p' options.
            p_table,
            _.THINGTYPE == p_table.name,
            how="inner",
        )
        .group_by("FILE_ID")
        .order_by("probability")
        .mutate(
            # We're making a discrete CDF here; we use that in the next three draw
            #   actions to identify which item corresponds to the sampled probability.
            cumulative_probability=_.probability.cumsum(),
        )
        .mutate(
            # DRAW!
            cum_p=_.p
            <= _.cumulative_probability
        )
        .filter(
            # DRAW!
            _.cum_p
        )
        .group_by("FILE_ID")
        .agg(
            # DRAW!
            next_step=_.next_step.argmin(_.cumulative_probability)
        )
    )
    return t


def single_step(
    data_layer: DataLayer,
    current_step: FileId,
    next_step: str,
    t_metadata: ibis.Table,
    rng,
) -> FileId:
    """Simulate the next step."""
    if next_step == "DrawData":
        f = _step_DrawData
    elif next_step == "ModelReady":
        f = _step_ModelReady
    elif next_step == "Model":
        f = _step_Model
    elif next_step == "Aggregate":
        f = _step_Aggregate
    elif next_step == "Review":
        f = _step_Review
    elif next_step == "Publish":
        f = _step_Publish
    else:
        raise RuntimeError
    return f(
        data_layer=data_layer, current_step=current_step, t_metadata=t_metadata, rng=rng
    )


def _step_DrawData(
    data_layer: DataLayer,
    current_step: FileId,
    t_metadata: ibis.Table,
    rng,
) -> FileId:
    """"""
    m = data_layer.get_metadata(current_step)
    p = data_layer.get_parameters(
        get_most_recent(
            t_metadata=t_metadata, thing_type="DrawData", project=m["PROJECT"]
        )
    )
    return data_layer.log(
        parameters=p,
        metadata={
            "TS_HAS_PARAMETERS": True,
            "TS_HAS_METADATA": True,
            "TS_HAS_DATASET": True,
            "TS_HAS_METRICS": True,
            "THINGTYPE": "DrawData",
            "JURISDICTION": m["JURISDICTION"],
            "PROJECT": m["PROJECT"],
        },
        # Notional empty raw data
        dataset=pd.DataFrame(),
        metrics={
            "dataset_size": 1000,
            "schema_length": 100,
            "creation_time": 30,
        },
    )


def _step_ModelReady(
    data_layer: DataLayer,
    current_step: FileId,
    t_metadata: ibis.Table,
    rng,
) -> FileId:
    """"""
    m = data_layer.get_metadata(current_step)
    dataset_fileid = get_most_recent(
        t_metadata=t_metadata, thing_type="DrawData", project=m["PROJECT"]
    )
    dataset_features = get_random_features(feature_map=feature_map, rng=rng)
    return data_layer.log(
        parameters={
            "dataset_fileid": dataset_fileid,
            "dataset_features": dataset_features,
        },
        metadata={
            "TS_HAS_PARAMETERS": True,
            "TS_HAS_METADATA": True,
            "TS_HAS_DATASET": True,
            "TS_HAS_METRICS": True,
            "THINGTYPE": "ModelReady",
            "JURISDICTION": m["JURISDICTION"],
            "PROJECT": m["PROJECT"],
        },
        # Notional empty model ready data
        dataset=pd.DataFrame(),
        metrics={
            "dataset_size": 1000,
            "schema_length": 10,
            "creation_time": 10,
        },
    )


def _step_Model(
    data_layer: DataLayer,
    current_step: FileId,
    t_metadata: ibis.Table,
    rng,
) -> FileId:
    """"""
    m = data_layer.get_metadata(current_step)
    dataset_fileid = get_most_recent(
        t_metadata=t_metadata, thing_type="ModelReady", project=m["PROJECT"]
    )
    dataset_features = data_layer.get_parameters(dataset_fileid)["dataset_features"]
    scores = [feature_map[feature_name] for feature_name in dataset_features]
    accuracy = np.mean(scores)
    precision = np.product(scores)
    recall = 1 - precision

    return data_layer.log(
        parameters={
            "modeling_data_fileid": dataset_fileid,
            "tree_depth": rng.choice(range(4, 9), 1)[0],
            "tree_nodes": rng.choice(range(15, 25), 1)[0],
            "split_criterion": rng.choice(["gini", "entropy"], 1)[0],
        },
        metadata={
            "TS_HAS_PARAMETERS": True,
            "TS_HAS_METADATA": True,
            "TS_HAS_DATASET": True,
            "TS_HAS_METRICS": True,
            "THINGTYPE": "Model",
            "JURISDICTION": m["JURISDICTION"],
            "PROJECT": m["PROJECT"],
        },
        # Notional empty labels to associate with input.
        dataset=pd.DataFrame(),
        metrics={
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        },
        artifacts={"function": "filepath_with_model_functional_artifacts"},
    )


def _step_Aggregate(
    data_layer: DataLayer,
    current_step: FileId,
    t_metadata: ibis.Table,
    rng,
) -> FileId:
    """"""
    m = data_layer.get_metadata(current_step)
    modeling_fileid = get_most_recent(
        t_metadata=t_metadata, thing_type="Model", project=m["PROJECT"]
    )
    dataset_fileid = data_layer.get_parameters(modeling_fileid)["modeling_data_fileid"]

    return data_layer.log(
        parameters={
            "modeling_output_fileid": modeling_fileid,
            "modeling_data_fileid": dataset_fileid,
            "agg_method": rng.choice(["weighted", "unweighted"], 1)[0],
        },
        metadata={
            "TS_HAS_PARAMETERS": True,
            "TS_HAS_METADATA": True,
            "TS_HAS_DATASET": True,
            "TS_HAS_METRICS": True,
            "THINGTYPE": "Aggregate",
            "JURISDICTION": m["JURISDICTION"],
            "PROJECT": m["PROJECT"],
        },
        # Notional empty labels to associate with input.
        dataset=pd.DataFrame(),
        metrics={
            "compression_ratio": np.abs(rng.normal(0, 0.3)),
        },
    )


def _step_Review(
    data_layer: DataLayer,
    current_step: FileId,
    t_metadata: ibis.Table,
    rng,
) -> FileId:
    """"""
    m = data_layer.get_metadata(current_step)
    aggregate_fileid = get_most_recent(
        t_metadata=t_metadata, thing_type="Aggregate", project=m["PROJECT"]
    )
    modeling_fileid = data_layer.get_parameters(aggregate_fileid)[
        "modeling_output_fileid"
    ]
    satisfaction = np.mean(list(data_layer.get_metrics(modeling_fileid).values()))
    return data_layer.log(
        parameters={
            "modeling_output_fileid": modeling_fileid,
            "aggregate_output_fileid": aggregate_fileid,
            "user": "Steve",
            "user_comment": rng.choice(
                ["Yay!", "Nay."], p=[satisfaction, 1 - satisfaction]
            ),
        },
        metadata={
            "TS_HAS_PARAMETERS": True,
            "TS_HAS_METADATA": True,
            "TS_HAS_DATASET": True,
            "TS_HAS_METRICS": True,
            "THINGTYPE": "Review",
            "JURISDICTION": m["JURISDICTION"],
            "PROJECT": m["PROJECT"],
        },
        metrics={
            "satisfaction": satisfaction,
        },
        artifacts={"review": "filepath_with_explicit_review_artifacts"},
    )


def _step_Publish(
    data_layer: DataLayer,
    current_step: FileId,
    t_metadata: ibis.Table,
    rng,
) -> FileId:
    """"""
    m = data_layer.get_metadata(current_step)
    review_fileid = get_most_recent(
        t_metadata=t_metadata, thing_type="Review", project=m["PROJECT"]
    )
    p = data_layer.get_parameters(review_fileid)
    met = data_layer.get_metrics(review_fileid)
    return data_layer.log(
        parameters={
            "modeling_output_fileid": p["modeling_output_fileid"],
            "aggregate_output_fileid": p["aggregate_output_fileid"],
            "review_output_fileid": review_fileid,
        },
        metadata={
            "TS_HAS_PARAMETERS": True,
            "TS_HAS_METADATA": True,
            "TS_HAS_DATASET": True,
            "TS_HAS_METRICS": True,
            "THINGTYPE": "Publish",
            "JURISDICTION": m["JURISDICTION"],
            "PROJECT": m["PROJECT"],
        },
        metrics={
            "meets_review": met["satisfaction"],
        },
        artifacts={"review": "filepath_with_explicit_review_artifacts"},
    )


def get_p_table():
    return (
        ibis.memtable(
            [{"name": k, "next_node": v["next_node"]} for k, v in functions.items()]
        )
        .unpack("next_node")
        .pivot_longer(~s.c("name"), names_to="next_step", values_to="probability")
        .drop_null()
        .execute()
    )


def get_most_recent(
    t_metadata: ibis.Table,
    thing_type: str,
    project: str,
) -> FileId:
    """"""
    x = t_metadata.filter(
        t_metadata.PROJECT == project,
        t_metadata.THINGTYPE == thing_type,
    )
    return x.FILE_ID.argmax(x.DATASET_DATE).execute()


def get_random_features(
    feature_map: Mapping[str, float], rng: Optional[Generator] = None
) -> List[str]:
    """"""
    return list(rng.choice(list(feature_map.keys()), 10))


def convenience_table(data_layer, t_metadata):
    return pd.concat(
        t_metadata.filter(_.PROJECT == -999)
        .group_by("FILE_ID")
        .mutate(latest=_.FILE_ID == _.FILE_ID.argmax(_.DATASET_DATE))
        .filter(_.latest)
        .order_by("DATASET_DATE")
        .mutate(i=ibis.row_number())
        .execute()
        .apply(lambda x: data_layer.load(x.FILE_ID).assign(i=x.i), axis=1)
        .to_list()
    ).rename(columns={"FILE_IDS": "FILE_ID"})


def vis_stepwise_thing_dist(data_layer):
    t_metadata = ibis.memtable(data_layer.browse())
    return (
        ggplot(
            t_metadata.select("FILE_ID", "THINGTYPE")
            .join(convenience_table(data_layer, t_metadata), "FILE_ID")
            .order_by(
                "i",
            )
            .group_by("i", "THINGTYPE")
            .agg(cnt=_.count())
            .pivot_wider(names_from="THINGTYPE", values_from="cnt")
            .fill_null(0)
            .order_by(
                "i",
            )
            .pivot_longer(~s.c("i"), names_to="THINGTYPE", values_to="cnt")
            .group_by("THINGTYPE")
            .order_by(
                "i",
            )
            .mutate(cnt=_.cnt.cumsum())
            .order_by("i", "THINGTYPE"),
            aes(x="i", y="cnt", fill="THINGTYPE"),
        )
        + geom_col()
        + theme(figure_size=(6, 4))
    )
