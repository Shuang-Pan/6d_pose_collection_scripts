from __future__ import annotations
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math
from pathlib import Path
from typing import Dict, Tuple
import cv2
import numpy as np
from enum import Enum
import yaml
import png
from ambf6dpose.DataCollection.DatasetSample import DatasetSample
from ambf6dpose.DataCollection.AbstractReaderSaver import AbstractSaver


class ImgDirs(Enum):
    RAW = "raw_img"
    SEGMENTED = "segmented_img"
    DEPTH = "depth_img"
    BLENDED = "blended_img"


class YamlFiles(Enum):
    EXTRINSIC = "extrinsic.yaml"
    INTRINSIC = "intrinsic.yaml"


class YamlKeys(Enum):
    ROT_EXT = "rot_model2cam"
    T_EXT = "t_model2cam"
    INTRINSIC = "intrinsic"


class DatasetConsts(Enum):
    NUM_OF_DIGITS_PER_STEP = 5
    MAX_STEP = math.pow(10, NUM_OF_DIGITS_PER_STEP) - 1
    FMT_STR = f"0{NUM_OF_DIGITS_PER_STEP}d"


@dataclass
class SampleSaver(AbstractSaver):
    img_saver: ImageSaver = field(default=None)
    yaml_saver: YamlSaver = field(default=None)

    def __post_init__(self):
        self.__max_step = DatasetConsts.MAX_STEP.value
        self.__internal_step = 0

        self.root.mkdir(exist_ok=True)

        if self.img_saver is None:
            self.img_saver = ImageSaver(self.root)

        if self.yaml_saver is None:
            self.yaml_saver = YamlSaver(self.root)

    @classmethod
    def fmt_step(cls, step) -> str:
        return f"{step:{DatasetConsts.FMT_STR.value}}"

    def save_sample(self, sample: DatasetSample):
        self.img_saver.save_sample(str_step=self.fmt_step(self.__internal_step), data=sample)
        self.yaml_saver.save_sample(str_step=self.fmt_step(self.__internal_step), data=sample)
        self.__internal_step += 1

        if self.__internal_step > self.__max_step:
            raise ValueError(
                f"Max number of samples reached. "
                "Modify `self.__num_of_digits_per_step` to collect bigger datasets "
                "than {self.__max_step}"
            )

    def close(self):
        self.yaml_saver.close()


@dataclass
class ImageSaver:
    root_path: Path

    def __post_init__(self):
        self.root_path.mkdir(exist_ok=True)
        self.dir_dict = self.get_img_dirs()
        self.create_dir()

    def get_img_dirs(self) -> Dict[ImgDirs, Path]:
        img_dirs = {}
        for imdir in ImgDirs:
            img_dirs[imdir] = self.root_path / imdir.value
        return img_dirs

    def create_dir(self):
        for dir in self.dir_dict.values():
            dir.mkdir(exist_ok=True)

    def save_sample(self, str_step: str, data: DatasetSample):
        raw_path = str(self.dir_dict[ImgDirs.RAW] / f"{str_step}.png")
        segmented_path = str(self.dir_dict[ImgDirs.SEGMENTED] / f"{str_step}.png")
        cv2.imwrite(raw_path, data.raw_img)
        cv2.imwrite(segmented_path, data.segmented_img)

        # Save depth
        self.save_depth(self.dir_dict[ImgDirs.DEPTH] / f"{str_step}.png", data.depth_img)

        data.generate_blended_img()
        blended_path = str(self.dir_dict[ImgDirs.BLENDED] / f"{str_step}.png")
        cv2.imwrite(blended_path, data.blended_img)
    
    def save_depth(self, path: str, depth_im:np.ndarray):
        path = str(path)
        if path.split('.')[-1].lower() != 'png':
            raise ValueError('Only PNG format is currently supported.')

        im_uint16 = np.round(depth_im).astype(np.uint16)

        # PyPNG library can save 16-bit PNG and is faster than imageio.imwrite().
        w_depth = png.Writer(depth_im.shape[1], depth_im.shape[0], greyscale=True, bitdepth=16)
        with open(path, 'wb') as f:
            w_depth.write(f, np.reshape(im_uint16, (-1, depth_im.shape[1])))


@dataclass
class NumpyStrFormatter:
    """
    Class to format numpy array into a string before writting the yaml file.
    """

    precision: int = 8
    max_line_width: int = 200

    def convert_to_yaml(self, data: np.ndarray) -> str:
        flat_data = data.ravel()
        str_repr = np.array2string(
            flat_data, separator=",", precision=self.precision, max_line_width=self.max_line_width
        )
        return str_repr[1:-1]

    def convert_from_yaml(self, data: str, shape: Tuple) -> np.ndarray:
        return np.fromstring(data, sep=",").reshape(shape)


@dataclass
class YamlSaver(ABC):
    root: Path

    def __post_init__(self):
        self.__first_sample = True
        self.numpy_fmt = NumpyStrFormatter()
        file_name = self.root / YamlFiles.EXTRINSIC.value
        if file_name.exists():
            msg = f"GT file: {file_name} already exists. Do you want to overwrite it? (y/n) "
            if input(msg) != "y":
                print("exiting ...")
                exit()

        self.file_handle = open(file_name, "wt", encoding="utf-8")

    def save_intrinsics(self, data: DatasetSample):
        intrinsics_str = self.numpy_fmt.convert_to_yaml(data.intrinsic_matrix)
        data_dict = {YamlKeys.INTRINSIC.value: intrinsics_str}

        with open(self.root / YamlFiles.INTRINSIC.value, "wt", encoding="utf-8") as file_handle:
            yaml.dump(
                data_dict,
                file_handle,
                Dumper=yaml.SafeDumper,
                default_style="",
                line_break=None,
                width=200,
            )

    def save_sample(self, str_step: str, data: DatasetSample):
        if self.__first_sample:
            self.save_intrinsics(data)
            self.__first_sample = False

        rot_str = self.numpy_fmt.convert_to_yaml(data.extrinsic_matrix[:3, :3])
        t_str = self.numpy_fmt.convert_to_yaml(data.extrinsic_matrix[:3, 3])
        data_dict = {f"{str_step}": {YamlKeys.ROT_EXT.value: rot_str, YamlKeys.T_EXT.value: t_str}}

        yaml.dump(
            data_dict,
            self.file_handle,
            Dumper=yaml.SafeDumper,
            default_style="",
            line_break=None,
            width=200,
        )
        self.file_handle.write("\n")
        self.file_handle.flush()

    def close(self):
        self.file_handle.close()