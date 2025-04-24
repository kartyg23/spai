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

from spai.models import backbones


class TestCLIPBackbone(unittest.TestCase):

    def test_forward(self) -> None:
        model = backbones.CLIPBackbone().cpu()
        image: torch.Tensor = torch.randn(4, 3, 224, 224).cpu()

        output: torch.Tensor = model(image)

        self.assertEqual(output.shape, (4, 12, 196, 768))
        self.assertEqual(image.dtype, output.dtype)


class TestDINOv2Backbone(unittest.TestCase):

    def test_forward(self) -> None:
        model = backbones.DINOv2Backbone().cpu()
        image: torch.Tensor = torch.randn(4, 3, 224, 224).cpu()

        output: torch.Tensor = model(image)

        self.assertEqual(output.shape, (4, 12, 256, 768))
        self.assertEqual(image.dtype, output.dtype)

    def test_forward_dinov2_large(self) -> None:
        model = backbones.DINOv2Backbone(dinov2_model="dinov2_vitl14").cpu()
        image: torch.Tensor = torch.randn(4, 3, 224, 224).cpu()

        output: torch.Tensor = model(image)

        self.assertEqual(output.shape, (4, 12, 256, 1024))
        self.assertEqual(image.dtype, output.dtype)

    def test_forward_dinov2_large_custom_intermediate_layers(self) -> None:
        intermediate_layers: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23)
        model = backbones.DINOv2Backbone(dinov2_model="dinov2_vitl14",
                                         intermediate_layers=intermediate_layers).cpu()
        image: torch.Tensor = torch.randn(4, 3, 224, 224).cpu()

        output: torch.Tensor = model(image)

        self.assertEqual(output.shape, (4, 12, 256, 1024))
        self.assertEqual(image.dtype, output.dtype)

    def test_forward_dinov2_grande(self) -> None:
        model = backbones.DINOv2Backbone(dinov2_model="dinov2_vitg14").cpu()
        image: torch.Tensor = torch.randn(4, 3, 224, 224).cpu()

        output: torch.Tensor = model(image)

        self.assertEqual(output.shape, (4, 12, 256, 1536))
        self.assertEqual(image.dtype, output.dtype)

    def test_forward_dinov2_grande_custom_intermediate_layers(self) -> None:
        intermediate_layers: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 39)
        model = backbones.DINOv2Backbone(dinov2_model="dinov2_vitg14",
                                         intermediate_layers=intermediate_layers).cpu()
        image: torch.Tensor = torch.randn(4, 3, 224, 224).cpu()

        output: torch.Tensor = model(image)

        self.assertEqual(output.shape, (4, 12, 256, 1536))
        self.assertEqual(image.dtype, output.dtype)
