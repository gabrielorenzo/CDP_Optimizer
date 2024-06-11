import os
import os.path as op

import cv2
import numpy as np
import skimage.io

from pipeline import Pipeline
from utils.yacs import Config


OUTPUT_DIR = './output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def demo_test_raw():
    cfg = Config('configs/test_onsemi.yaml')
    print(cfg)
    with cfg.unfreeze():
        cfg.awb.gr_gain='1000'
        print(type(cfg.awb.gr_gain))
    print(cfg.awb.gr_gain)
    pipeline = Pipeline(cfg)
    raw_path = 'raw/capture1.raw'
    bayer = np.fromfile(raw_path, dtype='uint16', sep='')
    print(bayer.shape)
    bayer = bayer.reshape((cfg.hardware.raw_height, cfg.hardware.raw_width))
    print(bayer.shape)
    data, _ = pipeline.execute(bayer)
    output_path = op.join(OUTPUT_DIR, 'capture1_awb.png') #save .png image, once an image is processed with an ISP it can't be saved to .raw
    output = cv2.cvtColor(data['output'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output)
    
    

def demo_nikon_d3x():
    cfg = Config('configs/nikon_d3x.yaml')
    pipeline = Pipeline(cfg)

    pgm_path = 'raw/color_checker.pgm'
    bayer = skimage.io.imread(pgm_path).astype(np.uint16)

    data, _ = pipeline.execute(bayer)

    output_path = op.join(OUTPUT_DIR, 'color_checker.png')
    output = cv2.cvtColor(data['output'], cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output)


def demo_intermediate_results():
    cfg = Config('configs/nikon_d3x.yaml')

    # Boost parameters for exaggeration effect
    with cfg.unfreeze():
        cfg.module_enable_status.nlm = True
        cfg.nlm.h = 15
        cfg.ceh.clip_limit = 0.04
        cfg.eeh.flat_threshold = 1
        cfg.eeh.flat_threshold = 2
        cfg.eeh.edge_gain = 1280
        cfg.hsc.saturation_gain = 320
        cfg.bcc.contrast_gain = 280

    pipeline = Pipeline(cfg)

    pgm_path = 'raw/color_checker.pgm'
    bayer = skimage.io.imread(pgm_path).astype(np.uint16)

    _, intermediates = pipeline.execute(bayer, save_intermediates=True)
    for module_name, result in intermediates.items():
        output = pipeline.get_output(result)
        output_path = op.join(OUTPUT_DIR, '{}.jpg'.format(module_name))
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output)


if __name__ == '__main__':
    print('Processing test raw...')
    demo_test_raw()
    #demo_intermediate_results()
