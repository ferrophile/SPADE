from data.base_dataset import BaseDataset, get_params, get_transform
import util.util as util

from PIL import Image
import os
import cv2
import torch
import numpy as np
from scipy import stats


def fetch_struct3d_path(root, sample, isfull, fn):
    scene, room, pos = sample
    path = '{}/2D_rendering/{}/perspective/{}/{}/{}.png'.format(
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

        paths = self.get_paths(opt)
        for i in range(len(paths)):
            util.natural_sort(paths[i])
            paths[i] = paths[i][:opt.max_dataset_size]

        self.label_paths = paths[0]
        self.instance_paths = paths[1]
        self.image_paths = paths[2]
        self.empty_image_paths = paths[3]
        self.layout_paths = paths[4]

        size = len(self.label_paths)
        self.dataset_size = size

        classes = np.genfromtxt('./struct3d_classes.csv', dtype='|U', delimiter=',')
        self.class_ids = {c[0]: i for (i, c) in enumerate(classes)}

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'

        split_list_path = os.path.join(opt.dataroot, '{}_split_transform.csv'.format(opt.phase))
        split_list = np.genfromtxt(split_list_path, delimiter=',', dtype='|U')
        label_paths = [fetch_struct3d_path(opt.dataroot, sample, 'full', 'semantic') for sample in split_list]
        instance_paths = [fetch_struct3d_path(opt.dataroot, sample, 'full', 'instance') for sample in split_list]
        image_paths = [fetch_struct3d_path(opt.dataroot, sample, 'full', 'rgb_rawlight') for sample in split_list]
        empty_paths = [fetch_struct3d_path(opt.dataroot, sample, 'empty', 'rgb_rawlight') for sample in split_list]
        layout_paths = [fetch_struct3d_path(opt.dataroot, sample, 'empty', 'layout') for sample in split_list]

        paths = [label_paths, instance_paths, image_paths, empty_paths, layout_paths]
        for i in range(len(paths)-1):
            assert len(paths[i]) == len(paths[i+1])

        return paths

    def get_label_onehot_tensor(self, label_tensor, instance_np):
        label_np = label_tensor.squeeze().numpy().astype(np.int)
        obj_ids = [i for i in np.unique(instance_np)]
        label_box_np = np.zeros((self.opt.semantic_nc, self.opt.crop_size, self.opt.crop_size), dtype=np.int)
        label_box_np[0] = 1
        label_fine_np = np.zeros((self.opt.semantic_nc, self.opt.crop_size, self.opt.crop_size), dtype=np.int)
        label_fine_np[0] = 1
        instance_label_np = np.zeros((self.opt.crop_size, self.opt.crop_size), dtype=np.int)

        for i, obj_id in enumerate(obj_ids):
            obj_mask = (instance_np == obj_id)
            instance_np[obj_mask] = i

            if obj_id == 65535:
                continue

            obj_class = int(stats.mode(label_np[obj_mask]).mode[0])
            obj_class = self.class_ids[str(obj_class)]

            rows = np.any(obj_mask, axis=1)
            cols = np.any(obj_mask, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            label_box_np[0, rmin:rmax + 1, cmin:cmax + 1] = 0
            label_box_np[obj_class, rmin:rmax + 1, cmin:cmax + 1] = 1

            label_fine_np[0][obj_mask] = 0
            label_fine_np[obj_class][obj_mask] = 1
            instance_label_np[obj_mask] = obj_class

        label_box_tensor = torch.Tensor(label_box_np)
        label_fine_tensor = torch.Tensor(label_fine_np)
        instance_label_tensor = torch.Tensor(instance_label_np)
        return label_box_tensor, label_fine_tensor, instance_label_tensor, instance_np

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor = torch.squeeze(label_tensor)

        instance_path = self.instance_paths[index]
        instance = Image.open(instance_path)
        transform_instance = get_transform(self.opt, params, method=Image.NEAREST, normalize=False, toTensor=False)
        instance_np = np.array(transform_instance(instance))

        label_box_tensor, label_fine_tensor, instance_label_tensor, instance_np = \
            self.get_label_onehot_tensor(label_tensor, instance_np)
        instance_tensor = torch.Tensor(instance_np)

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

        '''
        # normal
        layout_path = self.layout_paths[index]
        layout = Image.open(layout_path)
        layout = layout.convert('RGB')
        layout_tensor = transform_image(layout)
        '''
        
        input_dict = {
            'instance': instance_tensor,
            'semantic': instance_label_tensor,
            'label': label_box_tensor,
            'fine_label': label_fine_tensor,
            'image': image_tensor,
            'empty_image': empty_image_tensor,
            'path': image_path,
        }

        return input_dict

    def __len__(self):
        return self.dataset_size
