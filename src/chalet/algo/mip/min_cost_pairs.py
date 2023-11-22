# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixed Integer Programming (MIP) model/algorithm to minimize cost."""
import logging
import traceback
from typing import List, Set

import networkx as nx
import numpy as np
import pandas as pd
import xpress as xp

import chalet.algo.csp as csp
import chalet.algo.mip.helper as helper
import chalet.algo.util as util
from chalet.common.constants import ROUND_OFF_FACTOR
from chalet.log_config.config import set_mip_log_file
from chalet.model.processed_nodes import Nodes
from chalet.model.processed_od_pairs import OdPairs

logger = logging.getLogger(__name__)

DEMAND = "demand"
STATION = "station"


def min_cost_pairs(nodes, subgraphs, od_pairs, tol, max_run_time, log_dir):
    """Minimize cost of full OD pair demand coverage."""
    candidates, subgraph_indices, covered_demand, station_capacities = helper.get_subgraph_indices_and_candidates(
        od_pairs, nodes, subgraphs
    )

    if candidates.empty:
        return covered_demand, 0

    station_vars = xp.vars(candidates.index, name=STATION, vartype=xp.binary)
    model = _build_model(candidates, nodes, subgraphs, od_pairs, subgraph_indices, station_vars, log_dir)

    # fast heuristic for starting solution
    _construct_initial_solution(
        model, candidates, nodes, od_pairs, subgraph_indices, subgraphs, station_vars, station_capacities
    )

    station_sol = _set_model_attributes_and_solve(
        model,
        station_vars,
        subgraph_indices,
        od_pairs,
        nodes,
        subgraphs,
        candidates,
        max_run_time,
        tol,
        station_capacities,
    )

    covered_demand += od_pairs.loc[subgraph_indices, OdPairs.demand].sum()
    station_sol_arr = np.array(list(station_sol.items()))
    station_sol_arr = station_sol_arr[station_sol_arr[:, 1] > 0.5]
    nodes.loc[station_sol_arr[:, 0], Nodes.real] = True

    total_cost = util.remove_redundant_stations(nodes, subgraphs, od_pairs)

    return covered_demand, total_cost


def _build_model(candidates, nodes, subgraphs, od_pairs, subgraph_indices, station_vars, log_dir):
    logger.info("Building MIP model to minimize cost.")

    model = xp.problem()
    set_mip_log_file(model, log_dir)
    model.addVariable(station_vars)
    helper.initialize_separator_constraints(
        model, nodes, subgraphs, od_pairs, subgraph_indices, station_vars, pair_vars=None
    )

    # Set model objective to minimize cost
    objective = xp.Sum([station_vars[i] * nodes.at[i, Nodes.cost] for i in candidates.index])
    model.setObjective(objective, sense=xp.minimize)
    return model


def _construct_initial_solution(
    model, candidates, nodes, od_pairs, subgraph_indices, subgraphs, station_vars, station_capacities: pd.Series
):
    logger.info("Running heuristic for initial solution.")
    sol_set: Set = set()
    residual_station_capacities = station_capacities.copy()
    for k in subgraph_indices:
        demand = od_pairs.at[k, OdPairs.demand]
        excluded_nodes = list(residual_station_capacities.loc[residual_station_capacities < demand].index)
        path, path_cost = helper.get_cheapest_path(od_pairs, k, subgraphs, nodes, sol_set, excluded_nodes)
        for u in path:
            if util.is_station(u, nodes):  # adjust remaining capacity of used stations
                residual_station_capacities.at[u] -= demand
            if helper.is_candidate(u, nodes):  # save new nodes to solution
                sol_set.add(u)

    init_sol = util.remove_redundancy(sol_set, nodes, subgraphs, od_pairs)
    init_sol_vec = np.zeros(len(candidates))
    init_sol_index = [model.getIndex(station_vars[u]) for u in init_sol]
    init_sol_vec[init_sol_index] = 1
    init_cost = np.sum(
        [init_sol_vec[model.getIndex(station_vars[i])] * nodes.at[i, Nodes.cost] for i in candidates.index]
    )
    logger.info(f"Constructed initial solution. Cost = {init_cost}")
    model.addmipsol(init_sol_vec)


def _pre_check_int_sol(
    problem, model, station_vars, subgraph_indices, od_pairs, nodes, subgraphs, cutoff, station_capacities: pd.Series
):
    x: List = []
    problem.getlpsol(x, None, None, None)

    def sol_filter(u):
        return not helper.is_candidate(u, nodes) or x[model.getIndex(station_vars[u])] > 0.5

    residual_station_capacities = station_capacities.copy()
    for k in subgraph_indices:
        orig, dest = od_pairs.at[k, OdPairs.origin_id], od_pairs.at[k, OdPairs.destination_id]
        max_time, max_road_time = (
            od_pairs.at[k, OdPairs.max_time],
            od_pairs.at[k, OdPairs.max_road_time],
        )
        demand = od_pairs.at[k, OdPairs.demand]

        def filter_func(u):
            return sol_filter(u) and (not util.is_station(u, nodes) or residual_station_capacities.at[u] >= demand)

        path = csp.time_feasible_path(
            nx.subgraph_view(subgraphs[k], filter_node=filter_func),
            orig,
            dest,
            max_road_time,
            max_time,
        )

        if not path:
            return True, None

        for u in path:
            if util.is_station(u, nodes):  # adjust remaining capacity of used stations
                residual_station_capacities.at[u] -= demand

    return False, cutoff


def _set_model_attributes_and_solve(
    model, station_vars, subgraph_indices, od_pairs, nodes, subgraphs, candidates, max_run_time, tol, station_capacities
):
    bb_info = util.BranchAndBoundInfo(subgraph_indices)

    def separate_lazy_constraints(problem, data):
        try:
            return util.separate_lazy_constraints(
                problem,
                model,
                od_pairs,
                nodes,
                station_vars,
                candidates,
                subgraph_indices,
                subgraphs,
                bb_info,
                station_capacities,
            )
        except Exception:
            logger.error(f"Problem in callback: {traceback.format_exc()}")

    def pre_check_int_sol(problem, data, soltype, cutoff):
        try:
            if soltype == 0:  # if solution is found as optimal node relaxation, do not reject
                return False, cutoff
            return _pre_check_int_sol(
                problem, model, station_vars, subgraph_indices, od_pairs, nodes, subgraphs, cutoff, station_capacities
            )
        except Exception:
            logger.error(f"Problem in callback: {traceback.format_exc()}")

    model.addcbpreintsol(pre_check_int_sol, None)  # callback when integer solution is found
    model.addcboptnode(separate_lazy_constraints, None)  # callback for optimal node relaxation

    helper.set_model_controls(model, max_run_time, tol)

    logger.info("Starting MIP solver..")

    model.solve("d")  # solve with dual simplex

    logger.info("MIP solver finished.")
    logger.info(f"Added inequalities during callback: {bb_info.inequality_count}")
    logger.info(
        f"Total time spent in callbacks for separation: {round(bb_info.separation_time, ROUND_OFF_FACTOR)} secs."
    )
    logger.info(
        f"Total time spent in callbacks for primal heuristic: {round(bb_info.heuristic_time, ROUND_OFF_FACTOR)} secs."
    )
    station_sol = model.getSolution(station_vars)
    return station_sol
