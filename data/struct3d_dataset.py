from data.base_dataset import BaseDataset, get_params, get_transform
import util.util as util

from PIL import Image
import os
import cv2
import json
import torch
import numpy as np
from scipy import stats
from scipy.spatial.transform import Rotation as R


def fetch_struct3d_path(root, sample, isfull, fn):
    scene, room, pos = sample
    path = '{}/2D_rendering/{}/perspective/{}/{}/{}'.format(
        scene, room, isfull, pos, fn)
    path = os.path.join(root, path)
    return path


class Struct3DDataset(BaseDataset):

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = BaseDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(model='indoor')
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            parser.set_defaults(load_size=286)
        else:
            parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=29)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(no_pairing_check=True)
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, instance_paths, image_paths, empty_image_paths, pose_paths, boxes_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(instance_paths)
        util.natural_sort(image_paths)
        util.natural_sort(empty_image_paths)
        util.natural_sort(pose_paths)
        util.natural_sort(boxes_paths)

        self.label_paths = label_paths[:opt.max_dataset_size]
        self.instance_paths = instance_paths[:opt.max_dataset_size]
        self.image_paths = image_paths[:opt.max_dataset_size]
        self.empty_image_paths = empty_image_paths[:opt.max_dataset_size]
        self.pose_paths = pose_paths[:opt.max_dataset_size]
        self.boxes_paths = boxes_paths[:opt.max_dataset_size]
        size = len(self.label_paths)
        self.dataset_size = size

        classes = np.genfromtxt('./struct3d_classes.csv', dtype='|U', delimiter=',')
        self.class_ids = {c[0]: i for (i, c) in enumerate(classes)}

    def get_paths(self, opt):
        split_list_path = os.path.join(opt.dataroot, '{}_split.csv'.format(opt.phase))
        split_list = np.genfromtxt(split_list_path, delimiter=',', dtype='|U')
        label_paths = [fetch_struct3d_path(opt.dataroot, sample, 'full', 'semantic.png') for sample in split_list]
        instance_paths = [fetch_struct3d_path(opt.dataroot, sample, 'full', 'instance.png') for sample in split_list]
        image_paths = [fetch_struct3d_path(opt.dataroot, sample, 'full', 'rgb_rawlight.png') for sample in split_list]
        empty_paths = [fetch_struct3d_path(opt.dataroot, sample, 'empty', 'rgb_rawlight.png') for sample in split_list]
        pose_paths = [fetch_struct3d_path(opt.dataroot, sample, 'full', 'camera_pose.txt') for sample in split_list]

        boxes_paths = ['../detectron/proposals/{}_{}_{}.json'.format(*sample) for sample in split_list]

        assert len(image_paths) == len(label_paths) == len(empty_paths)
        return label_paths, instance_paths, image_paths, empty_paths, pose_paths, boxes_paths

    def get_label_onehot_tensor(self, label_tensor, instance_np):
        label_np = label_tensor.squeeze().numpy().astype(np.int)
        obj_ids = [i for i in np.unique(instance_np) if i != 65535]
        label_onehot_np = np.zeros((self.opt.semantic_nc, self.opt.crop_size, self.opt.crop_size), dtype=np.int)
        label_onehot_np[0] = 1

        for obj_id in obj_ids:
            obj_mask = (instance_np == obj_id)
            obj_class = int(stats.mode(label_np[obj_mask]).mode[0])
            obj_class = self.class_ids[str(obj_class)]

            rows = np.any(obj_mask, axis=1)
            cols = np.any(obj_mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            label_onehot_np[0, rmin:rmax + 1, cmin:cmax + 1] = 0
            label_onehot_np[obj_class, rmin:rmax + 1, cmin:cmax + 1] = 1

        label_onehot_tensor = torch.Tensor(label_onehot_np)
        return label_onehot_tensor

    def load_label_onehot_tensor(self, box_path, load_size):
        width, height = load_size
        label_onehot_np = np.zeros((self.opt.semantic_nc, height, width), dtype=np.int)
        with open(box_path, 'r') as f:
            anns = json.load(f)

        for box in anns:
            pred_class = box['class']
            x1, y1, x2, y2 = box['box']
            label_onehot_np[pred_class, y1:y2, x1:x2] = 1

        label_onehot_tensor = torch.Tensor(label_onehot_np)
        return label_onehot_tensor

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0

        instance_path = self.instance_paths[index]
        instance = Image.open(instance_path)
        transform_instance = get_transform(self.opt, params, method=Image.NEAREST, normalize=False, toTensor=False)
        instance_np = np.array(transform_instance(instance))

        # label_onehot_tensor = self.get_label_onehot_tensor(label_tensor, instance_np)
        label_onehot_tensor = self.load_label_onehot_tensor(self.boxes_paths[index], label.size)
        transform_label.transforms = transform_label.transforms[:1]
        label_onehot_tensor = transform_label(label_onehot_tensor)

        # full input image
        transform_image = get_transform(self.opt, params)

        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')
        image_tensor = transform_image(image)

        # empty input image
        empty_image_path = self.empty_image_paths[index]
        empty_image = Image.open(empty_image_path)
        empty_image = empty_image.convert('RGB')
        empty_image_tensor = transform_image(empty_image)

        # camera pose
        pose_path = self.pose_paths[index]
        pose_raw_np = np.genfromtxt(pose_path, delimiter=' ')
        # pose_np = np.zeros(6)
        # pose_np[:3] = pose_raw_np[:3]
        pose_np = np.zeros(3)

        rot_y = pose_raw_np[3:6]
        rot_z = pose_raw_np[6:9]
        rot_x = np.cross(rot_y, rot_z)
        rot = np.stack((rot_x, rot_y, rot_z)).T
        quat = R.from_matrix(rot).as_quat()
        pose_np = quat[1:] / np.linalg.norm(quat[1:]) * np.arccos(quat[0])
        pose_tensor = torch.Tensor(pose_np)
        
        input_dict = {
            'label': label_onehot_tensor,
            'image': image_tensor,
            'empty_image': empty_image_tensor,
            'pose': pose_tensor,
            'path': image_path
        }

        return input_dict

    def __len__(self):
        return self.dataset_size
