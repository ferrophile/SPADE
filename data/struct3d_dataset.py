from data.base_dataset import BaseDataset, get_params, get_transform
import util.util as util

from PIL import Image
import os
import cv2
import torch
import numpy as np


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
        parser.set_defaults(label_nc=40)
        parser.set_defaults(contain_dontcare_label=True)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        parser.set_defaults(no_instance=True)
        parser.set_defaults(no_pairing_check=True)
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths, empty_image_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        util.natural_sort(empty_image_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        empty_image_paths = empty_image_paths[:opt.max_dataset_size]

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.empty_image_paths = empty_image_paths

        size = len(self.label_paths)
        self.dataset_size = size

        object_names = np.genfromtxt('./nyu_classes.txt', dtype='|U14', delimiter=',')
        filter_names = ['wall', 'floor', 'otherstructure', 'otherfurniture', 'otherprop']
        self.object_list = [i for i, name in enumerate(object_names) if name not in filter_names]

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'

        split_list_path = os.path.join(opt.dataroot, '{}_split.csv'.format(opt.phase))
        split_list = np.genfromtxt(split_list_path, delimiter=',', dtype='|U')
        label_paths = [fetch_struct3d_path(opt.dataroot, sample, 'full', 'semantic') for sample in split_list]
        image_paths = [fetch_struct3d_path(opt.dataroot, sample, 'full', 'rgb_rawlight') for sample in split_list]
        empty_paths = [fetch_struct3d_path(opt.dataroot, sample, 'empty', 'rgb_rawlight') for sample in split_list]

        assert len(image_paths) == len(label_paths) == len(empty_paths)
        return label_paths, image_paths, empty_paths

    def get_onehot_box_tensor(self, label_tensor):
        label_tensor = label_tensor.view(label_tensor.size(1), label_tensor.size(2))
        label_np = label_tensor.data.cpu().numpy()
        label_np = label_np - 1

        save_label = np.zeros(label_np.shape)

        label_onehot = np.zeros((1, 41,) + label_np.shape)
        label_onehot_tensor = torch.from_numpy(label_onehot).float()
        label_seq = np.unique(label_np)

        for label_id in label_seq:
            if label_id not in self.object_list:
                continue

            temp_label = np.zeros(label_np.shape)
            temp_label[label_np==label_id] = label_id
            temp_label = temp_label.astype('uint8')
            contours, hierarchy = cv2.findContours(temp_label,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)[-2:]
            for conid in range(len(contours)):
                mask = np.zeros(label_np.shape, dtype="uint8") * 255
                cv2.drawContours(mask, contours, conid, int(label_id), -1)
                i, j = np.where(mask==label_id)
                indices = np.meshgrid(np.arange(min(i), max(i) + 1),
                                      np.arange(min(j), max(j) + 1),
                                      indexing='ij')
                x1 = min(i)
                y1 = min(j)
                x2 = max(i)
                y2 = max(j)
                if ((x2-x1) * (y2-y1)) < 30:
                    continue
                save_label[tuple(indices)] = label_id
                save_label_batch = save_label.reshape((1,1,)+save_label.shape)
                save_label_tensor = torch.from_numpy(save_label_batch).long()
                label_onehot_tensor.scatter_(1, save_label_tensor, 1.0)

        label_onehot_tensor = label_onehot_tensor.squeeze(0)
        return label_onehot_tensor, save_label

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_onehot_tensor, label_foreground = self.get_onehot_box_tensor(label_tensor)
        label_onehot_tensor = label_onehot_tensor.float()

        # Background
        fine_label_tensor = transform_label(label) * 255.0
        fine_label_tensor = fine_label_tensor - 1
        fine_label_tensor[fine_label_tensor == -1] = self.opt.label_nc

        label_back_tensor = np.zeros((1, 41, fine_label_tensor.size(1), fine_label_tensor.size(2)))
        label_back_tensor = torch.from_numpy(label_back_tensor).float()
        fine_label_tensor = fine_label_tensor.unsqueeze(0).long()
        label_back_tensor.scatter_(1, fine_label_tensor, 1.0)
        label_back_tensor = label_back_tensor.squeeze(0)

        # Foreground
        label_foreground[label_foreground!=0] = 1
        label_foreground = torch.from_numpy(label_foreground)
        label_back_tensor[:,label_foreground==1] = 0
        label_back_tensor[self.object_list] = label_onehot_tensor[self.object_list]
        label_back_tensor = label_back_tensor.float()

        # input image (real images)
        transform_image = get_transform(self.opt, params)

        image_path = self.image_paths[index]
        image = Image.open(image_path)
        image = image.convert('RGB')
        image_tensor = transform_image(image)

        empty_image_path = self.empty_image_paths[index]
        empty_image = Image.open(empty_image_path)
        empty_image = empty_image.convert('RGB')
        empty_image_tensor = transform_image(empty_image)
        
        input_dict = {
            'label': label_back_tensor,
            'image': image_tensor,
            'empty_image': empty_image_tensor,
            'path': image_path
        }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    # Change the 'unknown' label to the last label to match other datasets.
    def postprocess(self, input_dict):
        if len(input_dict['label']) == 1:
            label = input_dict['label']
            label = label - 1
            label[label == -1] = self.opt.label_nc

    def __len__(self):
        return self.dataset_size
