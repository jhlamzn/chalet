# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import chalet.algo.util as util
from tests.algo.util.util_data import UtilData

NODES = UtilData.nodes
SUB_GRAPHS = UtilData.sub_graphs
OD_PAIRS = UtilData.od_pairs
STATION_VARS = UtilData.station_vars
DEMAND_VARS = UtilData.demand_vars
CANDIDATES = UtilData.candidates
STATION_CAPACITIES = UtilData.station_capacities


def return_attribute(attrib: str):
    if attrib == "mipinfeas":
        return 1
    elif attrib == "currentnode":
        return 1
    elif attrib == "parentnode":
        return 0
    elif attrib == "mipobjval":
        return 1
    elif attrib == "mipbestobjval":
        return 1
    else:
        return None


class TestUtil:
    @patch.object(util.csp, "shortest_path")
    @patch.object(util.csp, "time_feasible_cheapest_path")
    @patch("networkx.dfs_preorder_nodes", return_value=0)
    @patch("networkx.node_boundary", return_value=[0])
    @patch("networkx.single_source_dijkstra_path_length", return_value={0: 0, 1: 1})
    def test_separate_lazy_constraints(
        self, mock_nx_algo, mock_nx_boundary, mock_nx_preorder, mock_cheapest_path, mock_shortest_path
    ):
        problem = Mock()
        model = Mock()

        def mock_lpsol(a, b, c, d):
            a.append(0)

        problem.getlpsol.side_effect = mock_lpsol
        model.getIndex.return_value = 0
        problem.getAttrib.side_effect = return_attribute
        subgraph_indices = [0]
        bb_info = util.BranchAndBoundInfo(subgraph_indices)
        mock_shortest_path.return_value = ([0, 1], 10.0)
        mock_cheapest_path.return_value = ([0, 1], 10.0)
        util.separate_lazy_constraints(
            problem,
            model,
            OD_PAIRS,
            NODES,
            STATION_VARS,
            CANDIDATES,
            subgraph_indices,
            SUB_GRAPHS,
            bb_info,
            STATION_CAPACITIES,
        )

        mock_shortest_path.assert_called()
        mock_cheapest_path.assert_called()
        mock_nx_algo.assert_called()
        mock_nx_preorder.assert_called()

    @patch.object(util, "_separation_algorithm")
    def test_separate_lazy_constraints_with_demand_vars(self, mock_separation_algo):
        mock_separation_algo.return_value = (1, [0])
        problem = Mock()
        model = Mock()

        def mock_lpsol(a, b, c, d):
            a.append(0)

        problem.getlpsol.side_effect = mock_lpsol
        model.getIndex.return_value = 0
        problem.getAttrib.side_effect = return_attribute
        subgraph_indices = [0]
        bb_info = util.BranchAndBoundInfo(subgraph_indices)
        util.separate_lazy_constraints(
            problem,
            model,
            OD_PAIRS,
            NODES,
            STATION_VARS,
            CANDIDATES,
            subgraph_indices,
            SUB_GRAPHS,
            bb_info,
            STATION_CAPACITIES,
            demand_vars=DEMAND_VARS,
        )

        mock_separation_algo.assert_called()

    @patch.object(util, "_separation_algorithm")
    def test_separate_lazy_constraints_with_solved_nodes(self, mock_separation_algo):
        problem = Mock()
        model = Mock()

        def mock_lpsol(a, b, c, d):
            a.append(0)

        problem.getlpsol.side_effect = mock_lpsol
        model.getIndex.return_value = 0
        problem.getAttrib.side_effect = return_attribute
        subgraph_indices = [0]
        bb_info = util.BranchAndBoundInfo(subgraph_indices)
        bb_info.frac_sep_rounds[1] = 100  # large enough so no further separation is called
        check = util.separate_lazy_constraints(
            problem,
            model,
            OD_PAIRS,
            NODES,
            STATION_VARS,
            CANDIDATES,
            subgraph_indices,
            SUB_GRAPHS,
            bb_info,
            STATION_CAPACITIES,
            demand_vars=DEMAND_VARS,
        )

        mock_separation_algo.assert_not_called()
        assert check is False

    @patch.object(util.csp, "shortest_path")
    @patch.object(util, "remove_redundancy")
    @patch.object(util, "_integer_time_separation")
    @patch("networkx.dfs_preorder_nodes", return_value=0)
    @patch("networkx.node_boundary", return_value=[0])
    def test_separate_lazy_constraints_without_shortest_path(
        self, mock_nx_boundary, mock_nx_preorder, mock_time_separation, mock_redundancy, mock_shortest_path
    ):
        problem = Mock()
        model = Mock()

        def mock_lpsol(a, b, c, d):
            a.append(0)

        problem.getlpsol.side_effect = mock_lpsol
        model.getIndex.return_value = 0
        problem.getAttrib.side_effect = return_attribute
        mock_shortest_path.return_value = ([], 1.0)
        mock_redundancy.return_value = [0]
        mock_time_separation.return_value = 1
        subgraph_indices = [0]
        bb_info = util.BranchAndBoundInfo(subgraph_indices)
        bb_info.frac_sep_rounds[1] = 0
        util.separate_lazy_constraints(
            problem,
            model,
            OD_PAIRS,
            NODES,
            STATION_VARS,
            CANDIDATES,
            subgraph_indices,
            SUB_GRAPHS,
            bb_info,
            STATION_CAPACITIES,
        )

        mock_shortest_path.assert_called()
        mock_nx_preorder.assert_called()

    @patch.object(util, "_add_inequality")
    @patch("networkx.dfs_preorder_nodes", return_value=0)
    @patch("networkx.node_boundary", return_value=[0])
    def test_integer_separation(self, mock_nx_boundary, mock_nx_preorder, mock_inequality):
        problem = Mock()
        separator = util._integer_separation(SUB_GRAPHS[0], 0, 1, True, 0, NODES, STATION_VARS, problem)

        assert separator == 1
        mock_nx_boundary.assert_called()
        mock_nx_preorder.assert_called()
