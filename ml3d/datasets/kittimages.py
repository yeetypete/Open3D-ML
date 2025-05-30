import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
from pathlib import Path
from glob import glob
import logging
import yaml
import open3d as o3d

from .base_dataset import BaseDataset, BaseDatasetSplit
from ..utils import Config, make_dir, DATASET
from .utils import DataProcessing, BEVBox3D
import pandas as pd

log = logging.getLogger(__name__)


class KITTImages(BaseDataset):
    """This class is used to create a dataset based on the KITTI dataset, and
    used in object detection, visualizer, training, or testing.
    """

    def __init__(self,
                 dataset_path,
                 name='KITTI',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 val_split=3712,
                 test_result_folder='./test',
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (KITTI in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            val_split: The split value to get a set of images for training,
            validation, for testing.
            test_result_folder: Path to store test output.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         val_split=val_split,
                         test_result_folder=test_result_folder,
                         **kwargs)

        cfg = self.cfg

        self.name = cfg.name
        self.dataset_path = cfg.dataset_path
        self.num_classes = 3
        self.label_to_names = self.get_label_to_names()

        self.train_info = []
        self.test_info = []
        self.val_info = []
        
        lidar_files = sorted(glob(join(cfg.dataset_path, 'training', 'velodyne', '*.bin')))
        image_files = sorted(glob(join(cfg.dataset_path, 'training', 'image_2', '*.png')))

        for i, (lidar, img) in enumerate(zip(lidar_files, image_files)):
            if i < cfg.val_split:
                self.train_info.append({'lidar_path': lidar,
                                      'cams': {'CAM2': {'data_path': img}}})
            else:
                self.val_info.append({'lidar_path': lidar,
                                      'cams': {'CAM2': {'data_path': img}}})
        
        lidar_files = sorted(glob(join(cfg.dataset_path, 'testing', 'velodyne', '*.bin')))
        image_files = sorted(glob(join(cfg.dataset_path, 'testing', 'image_2', '*.png')))
        
        for lidar, img in zip(lidar_files, image_files):
            self.test_info.append({'lidar_path': lidar,
                                   'cams': {'CAM2': {'data_path': img}}})

        self.all_files = glob(
            join(cfg.dataset_path, 'training', 'velodyne', '*.bin'))
        self.all_files.sort()
        self.train_files = []
        self.val_files = []

        for f in self.all_files:
            idx = int(Path(f).name.replace('.bin', ''))
            if idx < cfg.val_split:
                self.train_files.append(f)
            else:
                self.val_files.append(f)

        self.test_files = glob(
            join(cfg.dataset_path, 'testing', 'velodyne', '*.bin'))
        self.test_files.sort()

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and values are the corresponding
            names.
        """
        label_to_names = {
            0: 'Pedestrian',
            1: 'Cyclist',
            2: 'Car',
            3: 'Van',
            4: 'Person_sitting',
            5: 'Truck',
            6: 'Tram',
            7: 'Misc',
            8: 'DontCare'
        }
        return label_to_names

    @staticmethod
    def read_lidar(path):
        """Reads lidar data from the path provided.

        Returns:
            A data object with lidar information.
        """
        assert Path(path).exists()
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    
    @staticmethod
    def read_image(path):
        """Reads the left color image from the path provided.
        Returns:
            A numpy array representation of the image.
        """
        assert Path(path).exists(), "Path does not exist: {}".format(path)
        image = o3d.io.read_image(path)
        return np.array(image)

    @staticmethod
    def read_label(path, calib):
        """Reads labels of bound boxes.

        Returns:
            The data objects with bound boxes information.
        """
        def str_to_cov(s):
            values = list(map(float, s.split()))
            
            if len(values) != 6:
                raise ValueError("Input string must have 6 space-separated values.")
            
            a, b, c, d, e, f = values
            
            return np.array([[a, b, c],
                            [b, d, e],
                            [c, e, f]])
        
        if not Path(path).exists():
            return []

        with open(path, 'r') as f:
            lines = f.readlines()

        objects = []
        for line in lines:
            label = line.strip().split(' ')

            center = np.array(
                [float(label[11]),
                 float(label[12]),
                 float(label[13]), 1.0])

            points = center @ np.linalg.inv(calib['world_cam'])

            size = [float(label[9]), float(label[8]), float(label[10])]  # w,h,l
            center = [points[0], points[1], size[1] / 2 + points[2]]

            obj3d = Object3d(center, size, label, calib)
            if len(label) > 16: # add center covariance if available
                cov_str = ' '.join(label[16:22])
                center_cov = str_to_cov(cov_str)
                cam_world = np.linalg.inv(calib['world_cam'])
                center_cov = cam_world[:3,:3] @ center_cov @ cam_world[:3,:3].T
                obj3d.center_cov = center_cov
            if len(label) == 23: # add fusion
                rel_error = label[22]
                if rel_error == "-0":
                    obj3d.is_fusion = False
                    obj3d.rel_error = None
                else:
                    obj3d.is_fusion = True
                    obj3d.rel_error = float(rel_error)
            objects.append(obj3d)

        return objects

    @staticmethod
    def _extend_matrix(mat):
        mat = np.concatenate(
            [mat, np.array([[0., 0., 1., 0.]], dtype=mat.dtype)], axis=0)
        return mat

    @staticmethod
    def read_calib(path):
        """Reads calibiration for the dataset. You can use them to compare
        modeled results to observed results.

        Returns:
            The camera and the camera image used in calibration.
        """
        assert Path(path).exists()

        with open(path, 'r') as f:
            lines = f.readlines()

        obj = lines[0].strip().split(' ')[1:]
        P0 = np.array(obj, dtype=np.float32).reshape(3, 4)

        obj = lines[1].strip().split(' ')[1:]
        P1 = np.array(obj, dtype=np.float32).reshape(3, 4)

        obj = lines[2].strip().split(' ')[1:]
        P2 = np.array(obj, dtype=np.float32).reshape(3, 4)

        obj = lines[3].strip().split(' ')[1:]
        P3 = np.array(obj, dtype=np.float32).reshape(3, 4)

        P0 = KITTImages._extend_matrix(P0)
        P1 = KITTImages._extend_matrix(P1)
        P2 = KITTImages._extend_matrix(P2)
        P3 = KITTImages._extend_matrix(P3)

        obj = lines[4].strip().split(' ')[1:]
        rect_4x4 = np.eye(4, dtype=np.float32)
        rect_4x4[:3, :3] = np.array(obj, dtype=np.float32).reshape(3, 3)

        obj = lines[5].strip().split(' ')[1:]
        Tr_velo_to_cam = np.eye(4, dtype=np.float32)
        Tr_velo_to_cam[:3] = np.array(obj, dtype=np.float32).reshape(3, 4)

        world_cam = np.transpose(rect_4x4 @ Tr_velo_to_cam)
        cam_img = np.transpose(P2)

        return {'world_cam': world_cam, 'cam_img': cam_img}

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return KITTImagesSplit(self, split=split)

    def get_split_list(self, split):
        """Returns the list of data splits available.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The
            split name should be one of 'training', 'test', 'validation', or
            'all'.
        """
        if split in ['train', 'training']:
            return self.train_info
        elif split in ['test', 'testing']:
            return self.test_info
        elif split in ['val', 'validation']:
            return self.val_info
        raise ValueError("Invalid split {}".format(split))

    def is_tested(self):
        """Checks if a datum in the dataset has been tested.

        Args:
            dataset: The current dataset to which the datum belongs to.
            attr: The attribute that needs to be checked.

        Returns:
            If the dataum attribute is tested, then return the path where the
            attribute is stored; else, returns false.
        """
        pass

    def save_test_result(self, results, attrs):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the
            attribute passed.
            attrs: The attributes that correspond to the outputs passed in
            results.
        """
        make_dir(self.cfg.test_result_folder)
        for attr, res in zip(attrs, results):
            name = attr['name']
            path = join(self.cfg.test_result_folder, name + '.txt')
            f = open(path, 'w')
            for box in res:
                f.write(box.to_kitti_format(box.confidence))
                f.write('\n')


class KITTImagesSplit():

    def __init__(self, dataset, split='train'):
        self.cfg = dataset.cfg

        self.infos = dataset.get_split_list(split)
        self.path_list = []
        for info in self.infos:
            self.path_list.append(info['lidar_path'])

        log.info("Found {} pointclouds for {}".format(len(self.infos), split))
    
        self.split = split
        self.dataset = dataset

    def __len__(self):
        return len(self.infos)

    def get_data(self, idx):
        info = self.infos[idx]
        lidar_path = info['lidar_path']
        cam_path = info['cams']['CAM2']['data_path']
        label_path = lidar_path.replace('velodyne', 'label_2').replace('.bin', '.txt')
        calib_path = lidar_path.replace('velodyne', 'calib').replace('.bin', '.txt')

        pc = self.dataset.read_lidar(lidar_path)

        calib = self.dataset.read_calib(calib_path)
        label = self.dataset.read_label(label_path, calib)
        img = self.dataset.read_image(cam_path)
        lidar2cam_rt = calib['world_cam'].T
        lidar2img_rt = calib['cam_img'].T @ calib['world_cam'].T
        cam_intrinsic = calib['cam_img'].T

        cams = {'CAM2': {'img': img, 
                         'lidar2cam_rt': lidar2cam_rt,
                         'lidar2img_rt': lidar2img_rt,
                         'cam_intrinsic': cam_intrinsic}}
        
        img_shape = cams['CAM2']['img'].shape[:2]
        reduced_pc = DataProcessing.remove_outside_points(
            pc, calib['world_cam'], calib['cam_img'], img_shape)

        data = {
            'point': reduced_pc,
            'full_point': pc,
            'feat': None,
            'calib': calib,
            'bounding_boxes': label,
            'cams': cams,
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        name = Path(pc_path).name.split('.')[0]

        attr = {'name': name, 'path': pc_path, 'split': self.split}
        return attr

class Object3d(BEVBox3D):
    """The class stores details that are object-specific, such as bounding box
    coordinates, occulusion and so on.
    """

    def __init__(self, center, size, label, calib=None):
        confidence = float(label[15]) if label.__len__() >=16 else -1.0

        world_cam = calib['world_cam']
        cam_img = calib['cam_img']

        # kitti boxes are pointing backwards
        yaw = float(label[14]) - np.pi
        yaw = yaw - np.floor(yaw / (2 * np.pi) + 0.5) * 2 * np.pi

        self.truncation = float(label[1])
        self.occlusion = float(
            label[2]
        )  # 0:fully visible 1:partly occluded 2:largely occluded 3:unknown

        self.alpha = float(label[3])
        self.box2d = np.array((float(label[4]), float(label[5]), float(
            label[6]), float(label[7])),
                              dtype=np.float32)
        self.cam_name = 'CAM2'

        # class_name = label[0] if label[0] in KITTImages.get_label_to_names().values(
        # ) else 'DontCare'
        class_name = label[0]

        super().__init__(center, size, yaw, class_name, confidence, world_cam,
                         cam_img)

        self.yaw = float(label[14])

    def get_difficulty(self):
        """The method determines difficulty level of the object, such as Easy,
        Moderate, or Hard.
        """
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if height >= 40 and self.truncation <= 0.15 and self.occlusion <= 0:
            self.level_str = 'Easy'
            return 0  # Easy
        elif height >= 25 and self.truncation <= 0.3 and self.occlusion <= 1:
            self.level_str = 'Moderate'
            return 1  # Moderate
        elif height >= 25 and self.truncation <= 0.5 and self.occlusion <= 2:
            self.level_str = 'Hard'
            return 2  # Hard
        else:
            self.level_str = 'UnKnown'
            return -1

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.label_class, self.truncation, self.occlusion, self.alpha, self.box2d, self.size[2], self.size[0], self.size[1],
                        self.center, self.yaw)
        return print_str


DATASET._register_module(KITTImages)
