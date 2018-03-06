#!/usr/bin/python3 -B

import os

import glob
import numpy as np
import pandas as pd
from PIL import Image
from skimage.io import imread
from skimage.morphology import label
from skimage.transform import resize
from tqdm import tqdm

# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

# Constants.
IMG_CHANNELS = 3
PRED_DIR = '/home/ubuntu/workspace/kgg/bowl2018/code/pytorch-seg/results/'
TEST_PATH = '../input/stage1_test/'

# Get test IDs.
test_ids = next(os.walk(TEST_PATH))[1]

# Get size of test images.
sizes_test = []
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])

# Create list of upsampled test masks.
pred_files = glob.glob(os.path.join(PRED_DIR, '*.png'))
pred_files.sort()
preds_test_upsampled = []
for i, f in enumerate(pred_files):
    print(f)
    img = imread(f)
    preds_test_upsampled.append(resize(img,
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))

new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('20180225-sub-dsbowl2018-pytorchseg.csv', index=False)
