from data.image_folder import make_dataset
from data.base_dataset import BaseDataset, get_params, get_transform
import util.util as util
from PIL import Image

def _is_label(path):
    splits = path.split('/')
    return splits[-1] == 'semantic.png' and splits[-3] == 'full'

def _is_image(path):
    splits = path.split('/')
    return splits[-1] == 'rgb_rawlight.png' and splits[-3] == 'full'

def _is_empty_image(path):
    splits = path.split('/')
    return splits[-1] == 'rgb_rawlight.png' and splits[-3] == 'empty'

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
        self.empty_image_paths = image_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        root = opt.dataroot
        phase = 'val' if opt.phase == 'test' else 'train'
        all_images = make_dataset(root, recursive=True, read_cache=False, write_cache=False)
        label_paths = [path for path in all_images if _is_label(path)]
        image_paths = [path for path in all_images if _is_image(path)]
        empty_image_paths = [path for path in all_images if _is_empty_image(path)]
        assert len(image_paths) == len(label_paths) == len(empty_image_paths)

        instance_paths = []
        return label_paths, image_paths, empty_image_paths

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

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

        input_dict = {'label': label_tensor,
                      'image': image_tensor,
                      'empty_image': empty_image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    # In ADE20k, 'unknown' label is of value 0.
    # Change the 'unknown' label to the last label to match other datasets.
    def postprocess(self, input_dict):
        label = input_dict['label']
        label = label - 1
        label[label == -1] = self.opt.label_nc

    def __len__(self):
        return self.dataset_size
