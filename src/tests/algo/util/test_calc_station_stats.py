# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import ANY, call, patch

import networkx as nx
import pandas as pd
from pandas.testing import assert_frame_equal

from chalet.algo import util
from chalet.model.input.node_type import NodeType
from chalet.model.processed_arcs import Arcs
from chalet.model.processed_nodes import Nodes
from chalet.model.processed_od_pairs import OdPairs


def mock_get_path_attributes():
    return


@patch.object(util, "get_feasible_path", side_effect=[[1, 5, 6, 2], [2, 6, 7, 3], [3, 4]])
def test_calc_station_stats(patch_object):
    battery_capacity = 540
    terminal_range = 125.0
    truck_range = 300
    actual_nodes = pd.DataFrame(
        {
            Nodes.type: [
                NodeType.SITE,
                NodeType.SITE,
                NodeType.SITE,
                NodeType.SITE,
                NodeType.STATION,
                NodeType.STATION,
                NodeType.STATION,
                NodeType.STATION,
            ]
        },
        index=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    # calc_station_stats requires access to the following fields
    actual_nodes[Nodes.real] = [True, True, True, True, True, True, True, False]
    actual_nodes[Nodes.cost] = [0] * len(actual_nodes)
    actual_nodes[Nodes.capacity] = [float("inf")] * len(actual_nodes)

    expected_nodes = actual_nodes.copy()

    expected_nodes[Nodes.energy] = [0.0, 0.0, 0.0, 0.0, 180.0, 2790.0, 4860.0, 0.0]
    expected_nodes[Nodes.demand] = [0.0, 0.0, 0.0, 0.0, 10.0, 30.0, 20.0, 0.0]
    graph1 = nx.DiGraph()
    graph1.add_edges_from(
        [(1, 5), (5, 6), (6, 2)],
        **{Arcs.distance: 10.0},
        **{Arcs.time: 10.0},
        **{Arcs.fuel_time: 2.0},
        **{Arcs.break_time: 1.0}
    )
    graph2 = nx.DiGraph()
    graph2.add_edges_from(
        [(2, 6), (6, 7), (7, 3)],
        **{Arcs.distance: 10.0},
        **{Arcs.time: 10.0},
        **{Arcs.fuel_time: 2.0},
        **{Arcs.break_time: 1.0}
    )
    graph3 = nx.DiGraph()
    graph3.add_edge(
        3, 4, **{Arcs.distance: 10.0}, **{Arcs.time: 10.0}, **{Arcs.fuel_time: 2.0}, **{Arcs.break_time: 1.0}
    )
    subgraphs = [graph1, graph2, graph3]
    actual_od_pairs = pd.DataFrame(
        {OdPairs.origin_id: [1, 2, 3], OdPairs.destination_id: [2, 3, 4], OdPairs.demand: [10.0, 20.0, 30.0]}
    )
    expected_od_pairs = actual_od_pairs.copy()
    expected_od_pairs[OdPairs.stations] = ["5/6", "6/7", ""]
    expected_od_pairs[OdPairs.fuel_stops] = [2, 2, 0]
    expected_od_pairs[OdPairs.route_distance] = [30.0, 30.0, 10.0]
    expected_od_pairs[OdPairs.route_time] = [39.0, 39.0, 13.0]

    util.calc_station_stats(actual_nodes, subgraphs, actual_od_pairs, battery_capacity, terminal_range, truck_range)

    assert_frame_equal(actual_nodes, expected_nodes)
    assert_frame_equal(actual_od_pairs, expected_od_pairs)
    patch_object.assert_has_calls([call(subgraphs[i], i, actual_od_pairs, ANY) for i in range(3)])
