# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from pandas.testing import assert_series_equal

from chalet.model.input.node import Node
from chalet.model.parameters import Parameters
from chalet.model.processed_nodes import Nodes
from chalet.preprocess.nodes import PreprocessNodes


class TestPreprocessNodes:
    def test_preprocess(self):
        test_nodes = pd.DataFrame(
            {
                "ID": [0, 1, 2, 3],
                "TYPE": ["SITE", "SITE", "STATION", "STATION"],
                "COST": [0.0, 1.0, 1.0, 0.0],
            }
        )
        test_params = Parameters({"station_capacity": 500.0})
        test_data = {Node.get_file_name(): test_nodes, Parameters.get_file_name(): test_params}
        expected_real = pd.Series([True, False, False, True], name=Nodes.real)
        expected_capacity = pd.Series([500.0, 500.0, 500.0, 500.0], name=Nodes.capacity)
        pn = PreprocessNodes()

        pn.preprocess(test_data)
        actual_real = test_data[Node.get_file_name()][Nodes.real]
        actual_capacity = test_data[Node.get_file_name()][Nodes.capacity]

        assert_series_equal(expected_real, actual_real)
        assert_series_equal(expected_capacity, actual_capacity)
