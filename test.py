"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
# from models.pix2pix_model import Pix2PixModel
from models.indoor_model import IndoorModel
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()

dataloader = data.create_dataloader(opt)

model = IndoorModel(opt)
# model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)

# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

# test
for i, data_i in enumerate(dataloader):
    if i * opt.batchSize >= opt.how_many:
        break

    test_mode = 'transform_inference' if opt.transform_inference else 'inference'
    generated = model(data_i, mode=test_mode)

    img_path = data_i['path']
    for b in range(generated['input'].shape[0]):
        print('process image... %s' % img_path[b])

        if opt.transform_inference:
            visuals = OrderedDict([('input_label', generated['input'][b]),
                                   ('synthesized_label', generated['out_sem'][b]),
                                   ('fine_label', data_i['fine_label'][b]),
                                   ('empty_image', data_i['empty_image'][b])])
        else:
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('synthesized_image', generated['out_result'][b]),
                                   ('empty_image', data_i['empty_image'][b])])
        visualizer.save_images(webpage, visuals, img_path[b:b + 1])

webpage.save()
