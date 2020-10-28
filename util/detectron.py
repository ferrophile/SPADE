import copy
import json
import numpy as np
import os
import torch

from PIL import Image
from data.base_dataset import get_transform

from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, DatasetMapper
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.modeling import build_model
from detectron2.structures import BoxMode


class Struct3dMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        rgb_image = utils.read_image(dataset_dict['rgb_file_name'], format='RGB')
        sem_image = utils.read_image(dataset_dict['sem_file_name'], format='P')

        utils.check_image_size(dataset_dict, rgb_image)
        utils.check_image_size(dataset_dict, sem_image)

        aug_rgb_input = T.AugInput(rgb_image, sem_seg=None)
        aug_sem_input = T.AugInput(sem_image, sem_seg=None)
        rgb_image = aug_rgb_input.image
        sem_image = np.expand_dims(aug_sem_input.image, axis=2)

        images = []
        if 'RGB' in self.image_format:
            images.append(rgb_image)
        if 'P' in self.image_format:
            images.append(sem_image)
        image = np.concatenate(images, axis=2)

        transforms = self.augmentations(aug_rgb_input)

        image_shape = image.shape[:2]
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose((2, 0, 1))))

        if not self.is_train:
            dataset_dict.pop('annotations', None)
            return dataset_dict

        if 'annotations' in dataset_dict:
            for anno in dataset_dict['annotations']:
                if not self.use_instance_mask:
                    anno.pop('segmentation', None)
                if not self.use_keypoint:
                    anno.pop('keypoints', None)

            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop('annotations')
                if obj.get('iscrowd', 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict['instances'] = utils.filter_empty_instances(instances)

        return dataset_dict


class Struct3dTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference')

        evaluator = COCOEvaluator(dataset_name, cfg, True, output_folder)
        return evaluator

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = Struct3dMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = Struct3dMapper(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)


class Struct3dPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT

    def __call__(self, images):
        with torch.no_grad():
            original_image = np.concatenate(images, axis=2)
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype('float32').transpose((2, 0, 1)))

            inputs = {'image': image, 'height': height, 'width': width}
            predictions = self.model([inputs])[0]
            return predictions


def get_dataset(anns_path):
    with open(anns_path, 'r') as f:
        anns = json.load(f)

    for img in anns:
        for ann in img['annotations']:
            ann['bbox_mode'] = BoxMode(ann['bbox_mode'])

    return anns


class Struct3dDetectron:
    def __init__(self, ckpt_dir, mode='rgb_sem'):
        classes = np.genfromtxt('struct3d_classes.csv', delimiter=',', dtype='|U')
        self.nc = len(classes)
        self.classes_max_counts = classes[:, 2].astype(int)
        classes = classes[:, 1].tolist()

        DatasetCatalog.register('struct3d_train', lambda: get_dataset('train_anns.json'))
        DatasetCatalog.register('struct3d_test', lambda: get_dataset('test_anns.json'))
        MetadataCatalog.get('struct3d_train').thing_classes = classes
        MetadataCatalog.get('struct3d_test').thing_classes = classes

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
        cfg.DATASETS.TRAIN = ('struct3d_train', )
        cfg.DATASETS.TEST = ('struct3d_test', )
        cfg.INPUT.MIN_SIZE_TRAIN = (720, )
        cfg.INPUT.MAX_SIZE_TRAIN = 1280
        cfg.INPUT.MIN_SIZE_TEST = 720
        cfg.INPUT.MAX_SIZE_TEST = 1280
        cfg.MODEL.WEIGHTS = ""
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 30

        rgb_mean = cfg.MODEL.PIXEL_MEAN[::-1]
        rgb_std = cfg.MODEL.PIXEL_STD[::-1]
        pixel_mean = []
        pixel_std = []
        cfg.INPUT.FORMAT = ''
        if 'rgb' in mode:
            pixel_mean += rgb_mean
            pixel_std += rgb_std
            cfg.INPUT.FORMAT += 'RGB'
        if 'sem' in mode:
            pixel_mean += [15]
            pixel_std += [1.0]
            cfg.INPUT.FORMAT += 'P'

        cfg.MODEL.PIXEL_MEAN = pixel_mean
        cfg.MODEL.PIXEL_STD = pixel_std

        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.OUTPUT_DIR = ckpt_dir

        model = Struct3dTrainer.build_model(cfg)
        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=False)
        print('Weights "{}" loaded!'.format(cfg.MODEL.WEIGHTS))

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1

        self.cfg = cfg
        self.predictor = Struct3dPredictor(cfg)

    def predict(self, proposal_data, opt):
        assert proposal_data['rgb'].shape[0] == proposal_data['sem'].shape[0]

        transform_label = get_transform(opt, proposal_data['params'], method=Image.NEAREST, normalize=False, toTensor=False)

        labels = []
        for rgb_image, sem_image in zip(proposal_data['rgb'], proposal_data['sem']):
            print("Start prediction.")
            outputs = self.predictor([rgb_image, sem_image])['instances']

            counts = np.zeros_like(self.classes_max_counts)
            mask = np.ones_like(outputs.pred_classes.cpu(), dtype=bool)
            for i, pred in enumerate(outputs.pred_classes):
                counts[pred] += 1
                if counts[pred] > self.classes_max_counts[pred]:
                    mask[i] = False

            boxes_tensor = outputs.pred_boxes[mask].tensor.int()
            preds = outputs.pred_classes[mask]
            label_np = np.zeros((self.nc, *outputs.image_size))

            for box, pred in zip(boxes_tensor, preds):
                x1, y1, x2, y2 = box.cpu().numpy()
                pred = pred.cpu().numpy()
                label_np[pred, y1:y2, x1:x2] = 1

            labels.append(label_np)
            print('Predicted, pred_classes = ' + str(preds))

        labels_np = np.stack(labels, axis=0)
        transform_label.transforms = transform_label.transforms[:-1]
        labels_tensor = transform_label(torch.Tensor(labels_np))

        return labels_tensor
