# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Copyright (c) 2023 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
#
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.txt
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

"""
Created in September 2022
@author: fabrizio.guillaro
"""

import sys, os
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image

import torch
from torch.nn import functional as F

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
if path not in sys.path:
    sys.path.insert(0, path)

from config import update_config
from config import _C as config
from data_core import myDataset

parser = argparse.ArgumentParser(description='Test TruFor')
parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-in', '--input', type=str, default='../images',
                    help='can be a single file, a directory or a glob statement')
parser.add_argument('-out', '--output', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), '../outputs'), help='output folder')
parser.add_argument('-save_np', '--save_np', action='store_true', help='whether to save the Noiseprint++ or not')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()
update_config(config, args)

input = os.path.abspath(args.input)
output = os.path.abspath(args.output)
gpu = args.gpu
save_np = args.save_np

device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

if device != 'cpu':
    # cudnn setting
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

if '*' in input:
    list_img = glob(input, recursive=True)
    list_img = [img for img in list_img if not os.path.isdir(img)]
elif os.path.isfile(input):
    list_img = [input]
elif os.path.isdir(input):
    list_img = glob(os.path.join(input, '**/*'), recursive=True)
    list_img = [img for img in list_img if not os.path.isdir(img)]
else:
    raise ValueError("input is neither a file or a folder")

test_dataset = myDataset(list_img=list_img)

testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1)  # 1 to allow arbitrary input sizes

print('Input path: {}'.format(input))
print('Output path: {}'.format(output))
print('Save Noiseprint++: {}'.format(save_np))
print('Device: {}'.format(device))
print('Number of images found: {}'.format(len(list_img)))

if config.TEST.MODEL_FILE:
    model_state_file = config.TEST.MODEL_FILE
else:
    raise ValueError("Model file is not specified.")

print('=> loading model from {}'.format(model_state_file))
checkpoint = torch.load(model_state_file, map_location=torch.device(device), weights_only=False)

if config.MODEL.NAME == 'detconfcmx':
    from models.cmx.builder_np_conf import myEncoderDecoder as confcmx
    model = confcmx(cfg=config)
else:
    raise NotImplementedError('Model not implemented')

model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

with torch.no_grad():
    for index, (rgb, path) in enumerate(tqdm(testloader)):
        # filename_img = test_dataset.get_filename(index)

        path = os.path.abspath(path[0])

        if os.path.splitext(os.path.basename(output))[1] == '':  # output is a directory
            if '*' in input:
                root = os.path.abspath(input.split('*')[0])
            elif os.path.isfile(input):
                root = os.path.dirname(os.path.abspath(input))
            else:
                root = os.path.abspath(input)

            sub_path = os.path.relpath(path, root)
            filename_out = os.path.join(output, sub_path) + '.npz'
        else:  # output is a filename
            filename_out = output

        if os.path.splitext(filename_out)[1] == '':
            filename_out = filename_out + '.png'

        print('Processing image: {}'.format(path))
        print('Saving output to: {}'.format(filename_out))

        rgb = rgb.to(device)
        model.eval()

        det = None
        conf = None

        pred, conf, det, npp = model(rgb)
        print('Prediction shape: %s' % str(pred.shape))

        if conf is not None:
            conf = torch.squeeze(conf, 0)
            conf = torch.sigmoid(conf)[0]
            conf = conf.cpu().numpy()

        if npp is not None:
            npp = torch.squeeze(npp, 0)[0]
            npp = npp.cpu().numpy()

        if det is not None:
            det_sig = torch.sigmoid(det).item()

        pred = torch.squeeze(pred, 0)
        pred = F.softmax(pred, dim=0)[1]
        pred = pred.cpu().numpy()

        out_dict = dict()
        out_dict['map'] = pred
        out_dict['imgsize'] = tuple(rgb.shape[2:])
        if det is not None:
            out_dict['score'] = det_sig
        if conf is not None:
            out_dict['conf'] = conf
        if save_np:
            out_dict['np++'] = npp

        def norm_to_uint8(image):
            image = np.asarray(image, dtype=np.float32)
            if image.max() > 1.0 or image.min() < 0.0:
                image = image - image.min()
                denom = image.max() if image.max() != 0 else 1.0
                image = image / denom
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
            return image

        os.makedirs(os.path.dirname(filename_out), exist_ok=True)

        pred_image = norm_to_uint8(pred)
        Image.fromarray(pred_image, mode='L').save(filename_out[:-8] + '.png')

        if save_np and npp is not None:
            filename_out_np = os.path.splitext(filename_out)[0] + '_np.png'
            npp_image = norm_to_uint8(npp)
            Image.fromarray(npp_image, mode='L').save(filename_out_np)
            print('Also saved Noiseprint++ image to: {}'.format(filename_out_np))

