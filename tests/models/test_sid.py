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
from functools import partial
import random
from typing import Optional

import torch
from torch import nn

from spai.models import sid
from spai.models import vision_transformer
from spai.models import backbones


class TestPatchBasedMFViT(unittest.TestCase):

    def test_forward(self) -> None:
        batch_size: int = 4
        features_num: int = 12
        input_dim: int = 768
        masking_radius: int = 16

        backbone_vits = [
            vision_transformer.VisionTransformer(
                img_size=224,
                patch_size=16,
                in_chans=3,
                num_classes=2,
                embed_dim=768,
                depth=12,
                num_heads=12,
                mlp_ratio=4,
                qkv_bias=True,
                drop_rate=0.1,
                drop_path_rate=0.1,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=0.1,
                use_abs_pos_emb=True,
                use_rel_pos_bias=False,
                use_shared_rel_pos_bias=False,
                use_mean_pooling=False,
                use_intermediate_layers=True,
                intermediate_layers=tuple((i for i in range(12))),
                return_features=True
            ),
            backbones.CLIPBackbone().cpu(),
            backbones.DINOv2Backbone().cpu()
        ]

        for vit in backbone_vits:
            features_processor = sid.FrequencyRestorationEstimator(
                features_num=features_num,
                input_dim=input_dim,
                proj_dim=1024,
                proj_layers=2,
                patch_projection=True,
                patch_projection_per_feature=True
            )
            cls_head = sid.ClassificationHead(6 * features_num, 1, mlp_ratio=4)
            model = sid.PatchBasedMFViT(
                vit,
                features_processor,
                cls_head,
                masking_radius=masking_radius,
                img_patch_size=224,
                img_patch_stride=224,
                cls_vector_dim=6 * features_num,
                num_heads=8,
                attn_embed_dim=1536,
                dropout=0.2,
            )

            x: torch.Tensor = torch.randn((batch_size, 3, 448, 448))
            out: torch.Tensor = model(x)

            self.assertEqual(out.shape, torch.Size([batch_size, 1]))

    def test_forward_arbitrary_resolution(self) -> None:
        batch_size: int = 4
        features_num: int = 4
        input_dim: int = 768
        masking_radius: int = 16

        vit = vision_transformer.VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=2,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.1,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=0.1,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False,
            use_mean_pooling=False,
            use_intermediate_layers=True,
            intermediate_layers=(2, 5, 8, 11),
            return_features=True
        )
        features_processor = sid.FrequencyRestorationEstimator(
            features_num=features_num,
            input_dim=input_dim,
            proj_dim=1024,
            proj_layers=2,
            patch_projection=True,
            patch_projection_per_feature=True
        )
        cls_head = sid.ClassificationHead(6 * features_num, 1, mlp_ratio=4)
        model = sid.PatchBasedMFViT(
            vit,
            features_processor,
            cls_head,
            masking_radius=masking_radius,
            img_patch_size=224,
            img_patch_stride=224,
            cls_vector_dim=6 * features_num,
            num_heads=8,
            attn_embed_dim=1536,
            dropout=0.2,
        )

        x: list[torch.Tensor] = [
            torch.randn((1, 3, random.randint(224, 1000), random.randint(224, 1000)))
            for _ in range(batch_size)
        ]
        out: torch.Tensor = model(x, batch_size)

        self.assertEqual(out.shape, torch.Size([batch_size, 1]))


class TestMFViT(unittest.TestCase):

    def test_forward_pass(self) -> None:
        batch_size: int = 4
        features_num: int = 4
        input_dim: int = 768
        masking_radius: int = 16

        vit = vision_transformer.VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=2,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.1,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=0.1,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False,
            use_mean_pooling=False,
            use_intermediate_layers=True,
            intermediate_layers=(2, 5, 8, 11),
            return_features=True
        )
        features_processor = sid.FrequencyRestorationEstimator(
            features_num=features_num,
            input_dim=input_dim,
            proj_dim=1024,
            proj_layers=2,
            patch_projection=True,
            patch_projection_per_feature=True
        )
        cls_head = sid.ClassificationHead(6*features_num, 1, mlp_ratio=4)
        model = sid.MFViT(
            vit,
            features_processor,
            cls_head,
            masking_radius=masking_radius,
            img_size=224
        )

        x: torch.Tensor = torch.randn((batch_size, 3, 224, 224))
        out: torch.Tensor = model(x)

        self.assertEqual(out.shape, torch.Size([batch_size, 1]))


class TestFrequencyRestorationEstimator(unittest.TestCase):

    def test_forward_patch_projection_per_feature(self) -> None:
        batch_size: int = 4
        features_num: int = 4
        input_dim: int = 768
        proj_dim: int = 1024
        last_projection_layer_activation_types: list[Optional[str]] = ["gelu", None]
        use_original_image_feature_branch_cases: list[bool] = [True, False]

        x: torch.Tensor = torch.randn((batch_size, features_num, 196, input_dim))
        low_freq: torch.Tensor = torch.randn((batch_size, features_num, 196, input_dim))
        high_freq: torch.Tensor = torch.randn((batch_size, features_num, 196, input_dim))

        for proj_layer_activation_type in last_projection_layer_activation_types:
            for use_original_image_feature_branch in use_original_image_feature_branch_cases:
                model = sid.FrequencyRestorationEstimator(
                    features_num=features_num,
                    input_dim=input_dim,
                    proj_dim=proj_dim,
                    proj_layers=2,
                    patch_projection=True,
                    patch_projection_per_feature=True,
                    proj_last_layer_activation_type=proj_layer_activation_type,
                    original_image_features_branch=use_original_image_feature_branch
                )
                out: torch.Tensor = model(x, low_freq, high_freq)

                out_dim: int = 6 * features_num
                if use_original_image_feature_branch:
                    out_dim += proj_dim
                self.assertEqual(out.shape, torch.Size([batch_size, out_dim]))

    def test_forward_patch_projection_per_feature_without_reconstruction_similarity(self) -> None:
        batch_size: int = 4
        features_num: int = 4
        input_dim: int = 768
        proj_dim: int = 1024
        last_projection_layer_activation_types: list[Optional[str]] = ["gelu", None]
        use_original_image_feature_branch_cases: list[bool] = [True, False]

        x: torch.Tensor = torch.randn((batch_size, features_num, 196, input_dim))
        low_freq: torch.Tensor = torch.randn((batch_size, features_num, 196, input_dim))
        high_freq: torch.Tensor = torch.randn((batch_size, features_num, 196, input_dim))

        for proj_layer_activation_type in last_projection_layer_activation_types:
            for use_original_image_feature_branch in use_original_image_feature_branch_cases:
                raised_exception: bool = False
                try:
                    model = sid.FrequencyRestorationEstimator(
                        features_num=features_num,
                        input_dim=input_dim,
                        proj_dim=proj_dim,
                        proj_layers=2,
                        patch_projection=True,
                        patch_projection_per_feature=True,
                        proj_last_layer_activation_type=proj_layer_activation_type,
                        original_image_features_branch=use_original_image_feature_branch,
                        disable_reconstruction_similarity=True
                    )
                    out: torch.Tensor = model(x, low_freq, high_freq)

                    out_dim: int = proj_dim
                    self.assertEqual(out.shape, torch.Size([batch_size, out_dim]))
                except AssertionError:
                    raised_exception = True

                if not use_original_image_feature_branch:
                    self.assertTrue(raised_exception)
                else:
                    self.assertFalse(raised_exception)

class TestFeatureImportanceProjection(unittest.TestCase):

    def test_forward(self) -> None:
        batch_size: int = 4
        features_num: int = 4
        input_dim: int = 1024
        proj_dim: int = 1024

        x: torch.Tensor = torch.randn((batch_size, features_num, 196, input_dim))

        model = sid.FeatureImportanceProjector(
            features_num,
            input_dim,
            proj_dim,
            proj_layers=2,
        )

        out: torch.Tensor = model(x)
        self.assertEqual(out.shape, torch.Size([batch_size, proj_dim]))
