# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import networkx as nx
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from chalet.common.constants import OD_COVERAGE, SUB_GRAPHS, UNKNOWN_SITES
from chalet.model.input.od_pair import OdPair
from chalet.model.processed_arcs import Arcs
from chalet.model.processed_nodes import Nodes
from chalet.model.processed_od_pairs import OdPairs
from chalet.preprocess.od_pairs import PreprocessOdPairs
from tests.networkx_testing.testing import assert_graphs_equal
from tests.preprocess.stub_data import get_stub_data


def assert_data_equal(data1: Dict, data2: Dict):
    actual_od_pairs = data1[OdPair.get_file_name()]
    actual_unknown_sites = data1[UNKNOWN_SITES]
    actual_od_coverage = data1[OD_COVERAGE]
    actual_sub_graphs = data1[SUB_GRAPHS]

    expected_od_pairs = data2[OdPair.get_file_name()]
    expected_unknown_sites = data2[UNKNOWN_SITES]
    expected_od_coverage = data2[OD_COVERAGE]
    expected_sub_graphs = data2[SUB_GRAPHS]

    assert_frame_equal(actual_od_pairs, expected_od_pairs)
    assert_series_equal(actual_unknown_sites, expected_unknown_sites)
    assert_frame_equal(actual_od_coverage, expected_od_coverage)
    assert_graphs_equal(actual_sub_graphs, expected_sub_graphs)


class TestPreprocessOdPairs:
    """Test for PreprocessOdPairs class"""

    def test_preprocess(self):
        data = get_stub_data()
        pop = PreprocessOdPairs()
        graph1 = nx.DiGraph()
        graph1.add_nodes_from([1, 2], **{Nodes.cost: 0.0})
        graph1.add_edge(
            1,
            2,
            **{
                Arcs.time: 10.0,
                Arcs.fuel_time: 0.0,
                Arcs.break_time: 0.0,
                Arcs.distance: 10.0,
            }
        )
        expected_changed_data = {
            OdPair.get_file_name(): pd.DataFrame(
                {
                    OdPairs.origin_id: [3, 3, 2, 1],
                    OdPairs.destination_id: [4, 4, 3, 2],
                    OdPairs.demand: [40.0, 30.0, 20.0, 10.0],
                    OdPairs.distance: 4 * [10.0],
                    OdPairs.direct_time: 4 * [10.0],
                    OdPairs.max_time: 4 * [40.0],
                    OdPairs.max_road_time: 4 * [40.0],
                    OdPairs.feasible: [False, False, False, True],
                }
            ),
            UNKNOWN_SITES: pd.Series([5]),
            OD_COVERAGE: pd.DataFrame(
                {
                    OdPairs.origin_id: [3, 3, 2, 1],
                    OdPairs.destination_id: [4, 4, 3, 2],
                    OdPairs.demand: [40.0, 30.0, 20.0, 10.0],
                    OdPairs.distance: 4 * [10.0],
                    OdPairs.direct_time: 4 * [10.0],
                    OdPairs.max_time: 4 * [40.0],
                    OdPairs.max_road_time: 4 * [40.0],
                    OdPairs.feasible: [False, False, False, True],
                }
            ),
            SUB_GRAPHS: [nx.DiGraph(), nx.DiGraph(), nx.DiGraph(), graph1],
        }

        pop.preprocess(data)

        assert_data_equal(data, expected_changed_data)
