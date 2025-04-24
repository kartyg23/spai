# SPDX-FileCopyrightText: Copyright (c) 2025 Centre for Research and Technology Hellas
# and University of Amsterdam. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch

from spai.models import utils


class TestUtils(unittest.TestCase):

    def test_exportable_std(self) -> None:
        sizes: list[tuple[int, ...]] = [
            (5, 32, 1024),
            (1, 32, 1024),
            (5, 1, 1024),
        ]

        for s in sizes:
            x: torch.Tensor = torch.randn(s)
            pytorch_std: torch.Tensor = x.std(dim=-1)
            exportable_std: torch.Tensor = utils.exportable_std(x, dim=-1)
            self.assertTrue(torch.allclose(pytorch_std, exportable_std))
