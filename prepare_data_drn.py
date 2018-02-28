#!/usr/bin/python3 -B
# Script to prepare data to train Dilated Residual Network.
# cf. https://github.com/fyu/drn

import os
import random
import sys
import warnings

from itertools import chain
import json
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import numpy as np
from PIL import Image
from PIL import ImageEnhance
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
import tensorflow as tf
from tqdm import tqdm

from preprocess import central_scale_images, rotate_images, flip_images, \
    invert_images, add_salt_pepper_noise, add_gaussian_noise

# Set some parameters.
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'
DRN_PATH = '/home/ubuntu/workspace/kgg/bowl2018/input/drn/v1'
TRAIN_VAL_SPLIT = 0.9

for data_path in ['train', 'val', 'test']:
    if os.path.exists(os.path.join(DRN_PATH, data_path + '_images.txt')) or \
       os.path.exists(os.path.join(DRN_PATH, data_path + '_labels.txt')):
        print('Data list file exists.')
        raise FileNotFoundError
    for sub_path in ['rgb', 'gt']:
        if not os.path.exists(os.path.join(DRN_PATH, data_path, sub_path)):
            os.makedirs(os.path.join(DRN_PATH, data_path, sub_path))
# Delete "gt" directory in "test" as we don't have them.
os.rmdir(os.path.join(DRN_PATH, data_path, sub_path))

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train IDs.
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# Get and resize train and test datasets.
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')

# Data augmentation.
print('data augmentation')
print('scale images')
X_train_scale, Y_train_scale = central_scale_images(X_train, Y_train, [0.8, 0.4])

print('rotate images')
X_train_rot, Y_train_rot = rotate_images(X_train, Y_train)

print('flip images')
X_train_flip, Y_train_flip = flip_images(X_train, Y_train)

print('invert images')
X_train_inv, Y_train_inv = invert_images(X_train, Y_train)

print('add salt pepper noise')
X_train_sp, Y_train_sp = add_salt_pepper_noise(X_train, Y_train)

print('add gaussian noise')
X_train_gauss, Y_train_gauss = add_gaussian_noise(X_train, Y_train)

X_train = np.concatenate((X_train, X_train_scale, X_train_rot, X_train_flip, X_train_inv, X_train_sp, X_train_gauss))
Y_train = np.concatenate((Y_train, Y_train_scale, Y_train_rot, Y_train_flip, Y_train_inv, Y_train_sp, Y_train_gauss))

print('Compute mean and std values.')
mean_train = np.zeros(IMG_CHANNELS)
std_train = np.zeros(IMG_CHANNELS)
for x in tqdm(X_train, total=len(X_train)):
    mean_train += x.reshape(IMG_WIDTH * IMG_HEIGHT, IMG_CHANNELS).sum(axis=0).T
mean_train /= float(IMG_WIDTH * IMG_HEIGHT * len(X_train))
for x in tqdm(X_train, total=len(X_train)):
    std_train += np.absolute(x - mean_train.reshape([1, 1, 3])).reshape(IMG_WIDTH * IMG_HEIGHT, IMG_CHANNELS).sum(axis=0).T
std_train /= float(IMG_WIDTH * IMG_HEIGHT * len(X_train))
json_dict = {
    "mean": mean_train.tolist(),
    "std": std_train.tolist()
}
with open(os.path.join(DRN_PATH, "info.json"), 'w') as fw:
    fw.write(json.dumps(json_dict))

print('Make datasets.')
randids = np.random.permutation(len(X_train))
train_ids = randids[:int(len(X_train) * TRAIN_VAL_SPLIT)]
val_ids = randids[int(len(X_train) * TRAIN_VAL_SPLIT):]
print('Train')
for idx, rid in tqdm(enumerate(train_ids), total=len(train_ids)):
    x_pil = Image.fromarray(np.squeeze(X_train[rid]).astype(np.uint8))
    y_pil = Image.fromarray(np.squeeze(Y_train[rid]).astype(np.uint8))
    x_pil.save(os.path.join(DRN_PATH, 'train/rgb/{:08d}.png'.format(idx)))
    y_pil.save(os.path.join(DRN_PATH, 'train/gt/{:08d}.png'.format(idx)))
    with open(os.path.join(DRN_PATH, 'train_images.txt'), 'a') as fw:
        fw.write('train/rgb/{:08d}.png\n'.format(idx))
    with open(os.path.join(DRN_PATH, 'train_labels.txt'), 'a') as fw:
        fw.write('train/gt/{:08d}.png\n'.format(idx))
print('Val')
for idx, rid in tqdm(enumerate(val_ids), total=len(val_ids)):
    x_pil = Image.fromarray(np.squeeze(X_train[rid]).astype(np.uint8))
    y_pil = Image.fromarray(np.squeeze(Y_train[rid]).astype(np.uint8))
    x_pil.save(os.path.join(DRN_PATH, 'val/rgb/{:08d}.png'.format(idx)))
    y_pil.save(os.path.join(DRN_PATH, 'val/gt/{:08d}.png'.format(idx)))
    with open(os.path.join(DRN_PATH, 'val_images.txt'), 'a') as fw:
        fw.write('val/rgb/{:08d}.png\n'.format(idx))
    with open(os.path.join(DRN_PATH, 'val_labels.txt'), 'a') as fw:
        fw.write('val/gt/{:08d}.png\n'.format(idx))
print('Test')
for idx, x in tqdm(enumerate(X_test), total=len(X_test)):
    x_pil = Image.fromarray(np.squeeze(x).astype(np.uint8))
    x_pil.save(os.path.join(DRN_PATH, 'test/rgb/{:08d}.png'.format(idx)))
    with open(os.path.join(DRN_PATH, 'test_images.txt'), 'a') as fw:
        fw.write('test/rgb/{:08d}.png\n'.format(idx))
