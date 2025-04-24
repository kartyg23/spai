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

import itertools
import math
import unittest

import torch
import numpy as np
from PIL import Image

from spai.models import filters


class TestFilters(unittest.TestCase):

    def test_filter_image_frequencies(self, debug: bool = False) -> None:
        image_sizes: list[int] = [224, 384, 512, 1024]
        batch_sizes: list[int] = [1, 3, 8]
        mask_radiuses: list[int] = [16, 23]

        for image_size, batch_size, mask_radius in itertools.product(image_sizes, batch_sizes,
                                                                     mask_radiuses):

            image: torch.Tensor = torch.randn((batch_size, 3, image_size, image_size))
            mask: torch.Tensor = filters.generate_circular_mask(image_size, mask_radius)

            filtered: torch.Tensor
            res: torch.Tensor
            filtered, res = filters.filter_image_frequencies(image, mask)

            self.assertTrue((filtered+res-image).sum()/len(image)<1e-3)

            if debug and image_size == 1024 and batch_size == 1 and mask_radius == 23:
                Image.fromarray(
                    (image.squeeze(dim=0).permute((1, 2, 0)).numpy() * 255).astype(np.uint8)).show()
                Image.fromarray(
                    (filtered.squeeze(dim=0).permute((1, 2, 0)).numpy()*255).astype(
                        np.uint8)).show()
                Image.fromarray(
                    (res.squeeze(dim=0).permute((1, 2, 0)).numpy()*255).astype(np.uint8)).show()
                Image.fromarray(
                    ((filtered+res-image).squeeze(dim=0).permute((1, 2, 0)).numpy() * 255).astype(
                        np.uint8)).show()


    def test_generate_circular_mask_centered(self, debug: bool = False) -> None:
        input_sizes: list[int] = [224, 384, 512, 1024]
        mask_radiuses: list[int] = [16, 22, 30, 70]

        for input_size in input_sizes:
            for mask_radius in mask_radiuses:
                mask: torch.Tensor = filters.generate_circular_mask(input_size, mask_radius)
                self.assertEqual(mask.shape, (input_size, input_size))

                # Check that a rectangular area is properly masked.
                self.assertTrue(torch.all(mask[:input_size//2-mask_radius-1, :] == 0))
                self.assertTrue(torch.all(mask[input_size//2+mask_radius+1:, :] == 0))
                self.assertTrue(torch.all(mask[:, :input_size//2-mask_radius-1] == 0))
                self.assertTrue(torch.all(mask[:, input_size//2+mask_radius+1:] == 0))

                # Check that the inscribed rectangular area is not masked.
                sq_radius: int = int(mask_radius*math.sqrt(2)/2)
                self.assertTrue(torch.all(
                    mask[input_size//2-sq_radius:input_size//2+sq_radius+1,
                         input_size//2-sq_radius:input_size//2+sq_radius+1] == 1
                ))

                # DEBUG: Visualize the mask.
                if debug and input_size == 1024 and mask_radius == 70:
                    Image.fromarray((mask.numpy()*240+15).astype(np.uint8)).show()

    def test_generate_circular_mask_range(self, debug: bool = False) -> None:
        input_sizes: list[int] = [224, 384, 512, 1024]
        start_radiuses: list[int] = [16, 22, 30, 70]
        lengths: list[int] = [1, 2, 5, 14, 28]

        for input_size in input_sizes:
            for start_radius in start_radiuses:
                for length in lengths:
                    mask: torch.Tensor = filters.generate_circular_mask(
                        input_size, start_radius, start_radius+length
                    )
                    self.assertEqual(mask.shape, (input_size, input_size))

                    max_radius: int = start_radius+length

                    # Check that the rectangular area outside of mask range is not masked.
                    self.assertTrue(torch.all(mask[:input_size//2-max_radius-1, :] == 1))
                    self.assertTrue(torch.all(mask[input_size//2+max_radius+1:, :] == 1))
                    self.assertTrue(torch.all(mask[:, :input_size//2-max_radius-1] == 1))
                    self.assertTrue(torch.all(mask[:, input_size//2+max_radius+1:] == 1))

                    # Check that the inscribed rectangular area is not masked.
                    sq_radius: int = int(start_radius*math.sqrt(2)/2)
                    self.assertTrue(torch.all(
                        mask[input_size//2-sq_radius:input_size//2+sq_radius+1,
                             input_size//2-sq_radius:input_size//2+sq_radius+1] == 1
                    ))

                    # Check that some pixels in-between are masked.
                    self.assertTrue(torch.all(
                        mask[input_size//2-max_radius:input_size//2-start_radius-1,
                        input_size//2] == 0
                    ))

                    # DEBUG: Visualize the mask.
                    if debug and input_size == 1024 and start_radius == 70:
                        Image.fromarray((mask.numpy()*240+15).astype(np.uint8)).show()
