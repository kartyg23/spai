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
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional
from collections.abc import Callable

import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from torchvision.transforms.functional import ten_crop
from PIL import Image

from spai import data_utils
from spai.data import filestorage
from spai.data.data_finetune import CSVDataset, CSVDatasetTriplet


file_dir: Path = Path(__file__).parent


class TestCSVDataset(unittest.TestCase):

    def test_load_dataset(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_dir_path: Path = Path(temp_dir)

            classes_num: int = 4
            images_per_class: int = 10
            image_size: tuple[int, int] = (256, 256)
            splits: tuple[str, ...] = ("train", "val", "test")
            dataset_csv: Path = generate_random_dataset(
                temp_dir_path,
                classes_num=classes_num,
                images_per_class=images_per_class,
                image_size=image_size,
                splits=splits
            )
            lmdb_storage: Path = temp_dir_path / "temp_filestorage.lmdb"
            put_dataset_in_lmdb(dataset_csv, dataset_csv.parent, lmdb_storage)

            test_lmdb_paths: list[Optional[Path]] = [None, lmdb_storage]
            views_cases: list[int] = [1, 3, 4, 8]
            concatenate_views_cases: list[bool] = [False, True]

            for lmdb_path in test_lmdb_paths:
                for split in splits:
                    for views in views_cases:
                        for concatenate_views in concatenate_views_cases:
                            transform = A.Compose([ToTensorV2()])
                            dataset: CSVDataset = CSVDataset(
                                dataset_csv,
                                dataset_csv.parent,
                                split,
                                transform=transform,
                                lmdb_storage=lmdb_path,
                                views=views,
                                concatenate_views_horizontally=concatenate_views
                            )
                            self.assertEqual(len(dataset), classes_num*images_per_class)
                            self.assertEqual(
                                dataset.get_dataset_root_path(),
                                lmdb_path if lmdb_path else dataset_csv.parent
                            )

                            for i in range(len(dataset)):
                                item: tuple[torch.Tensor, np.ndarray, int] = dataset[i]
                                self.assertEqual(len(item), 3)
                                if concatenate_views:
                                    self.assertEqual(
                                        item[0].shape,
                                        (1, 3, image_size[0], views*image_size[1])
                                    )
                                else:
                                    self.assertEqual(
                                        item[0].shape,
                                        (views, 3, image_size[0], image_size[1])
                                    )

    def test_load_dataset_with_views_generator(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_dir_path: Path = Path(temp_dir)

            classes_num: int = 4
            images_per_class: int = 10
            image_size: tuple[int, int] = (256, 256)
            view_size: int = 224
            splits: tuple[str, ...] = ("train", "val", "test")
            views_generator: Callable[[Image.Image], tuple[Image.Image, ...]] = partial(
                ten_crop, size=224
            )

            dataset_csv: Path = generate_random_dataset(
                temp_dir_path,
                classes_num=classes_num,
                images_per_class=images_per_class,
                image_size=image_size,
                splits=splits,
            )
            lmdb_storage: Path = temp_dir_path / "temp_filestorage.lmdb"
            put_dataset_in_lmdb(dataset_csv, dataset_csv.parent, lmdb_storage)

            test_lmdb_paths: list[Optional[Path]] = [None, lmdb_storage]
            views_cases: list[int] = [1, 3, 4, 8]
            concatenate_views_cases: list[bool] = [False, True]

            for lmdb_path in test_lmdb_paths:
                for split in splits:
                    for views in views_cases:
                        for concatenate_views in concatenate_views_cases:
                            transform = A.Compose([ToTensorV2()])
                            dataset: CSVDataset = CSVDataset(
                                dataset_csv,
                                dataset_csv.parent,
                                split,
                                transform=transform,
                                lmdb_storage=lmdb_path,
                                views=views,
                                concatenate_views_horizontally=concatenate_views,
                                views_generator=views_generator
                            )
                            self.assertEqual(len(dataset), classes_num*images_per_class)
                            self.assertEqual(
                                dataset.get_dataset_root_path(),
                                lmdb_path if lmdb_path else dataset_csv.parent
                            )

                            for i in range(len(dataset)):
                                item: tuple[torch.Tensor, np.ndarray, int] = dataset[i]
                                self.assertEqual(len(item), 3)
                                if concatenate_views:
                                    self.assertEqual(
                                        item[0].shape,
                                        (1, 3, view_size, 10*view_size)
                                    )
                                else:
                                    self.assertEqual(
                                        item[0].shape,
                                        (10, 3, view_size, view_size)
                                    )


class TestCSVDatasetTriplet(unittest.TestCase):

    def test_load_dataset(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_dir_path: Path = Path(temp_dir)

            classes_num: int = 4
            images_per_class: int = 10
            image_size: tuple[int, int] = (256, 256)
            splits: tuple[str, ...] = ("train", "val", "test")
            dataset_csv: Path = generate_random_dataset(
                temp_dir_path,
                classes_num=classes_num,
                images_per_class=images_per_class,
                image_size=image_size,
                splits=splits
            )
            lmdb_storage: Path = temp_dir_path / "temp_filestorage.lmdb"
            put_dataset_in_lmdb(dataset_csv, dataset_csv.parent, lmdb_storage)

            test_lmdb_paths: list[Optional[Path]] = [None, lmdb_storage]

            for lmdb_path in test_lmdb_paths:
                for split in splits:
                    transform = A.Compose([ToTensorV2()])
                    dataset: CSVDatasetTriplet = CSVDatasetTriplet(
                        dataset_csv,
                        dataset_csv.parent,
                        split,
                        transform=transform,
                        lmdb_storage=lmdb_path,
                    )
                    self.assertEqual(len(dataset), classes_num*images_per_class)
                    self.assertEqual(
                        dataset.get_dataset_root_path(),
                        lmdb_path if lmdb_path else dataset_csv.parent
                    )

                    for i in range(len(dataset)):
                        item: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = dataset[i]
                        self.assertEqual(len(item), 3)
                        self.assertEqual(item[0].shape, (3, image_size[0], image_size[1]))
                        self.assertEqual(item[1].shape, (3, image_size[0], image_size[1]))
                        self.assertEqual(item[2].shape, (3, image_size[0], image_size[1]))


def generate_random_dataset(
    temp_dir: Path,
    classes_num: int = 4,
    images_per_class: int = 10,
    image_size: tuple[int, int] = (256, 256),
    splits: tuple[str, ...] = ("train", "val", "test")
) -> Path:
    entries: list[dict[str, Any]] = []

    # Generate random images
    for split in splits:
        for class_id in range(classes_num):
            for i in range(images_per_class):
                img_arr: np.ndarray = np.random.randint(
                    0, 256, size=(image_size[0], image_size[1], 3),
                    dtype=np.uint8
                )
                img_path: Path = temp_dir / f"class_{class_id}_image_{i}.jpg"
                Image.fromarray(img_arr).save(img_path)
                entries.append({
                    "image": str(img_path.relative_to(temp_dir)),
                    "class": class_id,
                    "split": split
                })

    # Export a temp csv.
    csv_path: Path = temp_dir / f"temp_dataset.csv"
    data_utils.write_csv_file(entries, csv_path, delimiter=",")

    return csv_path


def put_dataset_in_lmdb(
    dataset_csv: Path,
    csv_root_dir: Path,
    output_lmdb: Path
) -> None:
    filestorage.add_csv([
        "-c", str(dataset_csv.absolute()),
        "-b", str(csv_root_dir.absolute()),
        "-o", str(output_lmdb.absolute())
    ], standalone_mode=False)
