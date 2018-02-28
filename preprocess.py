# Ref:
#   - https://medium.com/ymedialabs-innovation/data-augmentation-techniques-in-cnn-using-tensorflow-371ae43d5be9
#   - https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277/notebook

import cv2
import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np


def central_scale_images(X_imgs, Y_imgs, scales):
    n ,h, w, c = X_imgs.shape

    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([h, w], dtype = np.int32)

    X_scale_data = []
    Y_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, h, w, 3))
    Y = tf.placeholder(tf.float32, shape = (1, h, w, 1))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_X_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    tf_Y_img = tf.image.crop_and_resize(Y, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for img_data, mask_data in zip(X_imgs, Y_imgs):
            batch_img = np.expand_dims(img_data, axis = 0)
            batch_mask = np.expand_dims(mask_data, axis = 0)
            scaled_imgs = sess.run(tf_X_img, feed_dict = {X: batch_img})
            scaled_masks = sess.run(tf_Y_img, feed_dict = {Y: batch_mask})
            X_scale_data.extend(scaled_imgs)
            Y_scale_data.extend(scaled_masks)

    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    Y_scale_data = np.array(Y_scale_data, dtype = np.float32)
    return X_scale_data, Y_scale_data


def rotate_images(X_imgs, Y_imgs):
    n ,h, w, c = X_imgs.shape

    X_rotate = []
    Y_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (h, w, 3))
    Y = tf.placeholder(tf.float32, shape = (h, w, 1))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    tf_mask = tf.image.rot90(Y, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img, mask in zip(X_imgs, Y_imgs):
            i = np.random.randint(3)  # Rotation at 90, 180, or 270.
            rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
            rotated_mask = sess.run(tf_mask, feed_dict = {Y: mask, k: i + 1})
            X_rotate.append(rotated_img)
            Y_rotate.append(rotated_mask)

    X_rotate = np.array(X_rotate, dtype = np.float32)
    Y_rotate = np.array(Y_rotate, dtype = np.float32)
    return X_rotate, Y_rotate


def flip_images(X_imgs, Y_imgs):
    n ,h, w, c = X_imgs.shape

    X_flip = []
    Y_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (h, w, 3))
    Y = tf.placeholder(tf.float32, shape = (h, w, 1))
    tf_imgX1 = tf.image.flip_left_right(X)
    tf_imgX2 = tf.image.flip_up_down(X)
    tf_imgX3 = tf.image.transpose_image(X)
    tf_imgY1 = tf.image.flip_left_right(Y)
    tf_imgY2 = tf.image.flip_up_down(Y)
    tf_imgY3 = tf.image.transpose_image(Y)
    tf_imgX = [tf_imgX1, tf_imgX2, tf_imgX3]
    tf_imgY = [tf_imgY1, tf_imgY2, tf_imgY3]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img, mask in zip(X_imgs, Y_imgs):
            i = np.random.randint(3)
            flipped_img = sess.run(tf_imgX[i], feed_dict = {X: img})
            flipped_mask = sess.run(tf_imgY[i], feed_dict = {Y: mask})
            X_flip.append(flipped_img)
            Y_flip.append(flipped_mask)
    X_flip = np.array(X_flip, dtype = np.float32)
    Y_flip = np.array(Y_flip, dtype = np.float32)
    return X_flip , Y_flip


def invert_images(X_imgs, Y_imgs):
    X_inv = []
    Y_inv = Y_imgs
    for X in X_imgs.astype(np.uint8):
        X_inv.append(np.invert(X))
    X_inv = np.array(X_inv, dtype=np.float32)
    return X_inv, Y_inv


def add_salt_pepper_noise(X_imgs, Y_imgs):
    # Need to produce a copy as to not modify the original image
    X_sp = X_imgs.copy()
    Y_sp = Y_imgs
    row, col, _ = X_sp[0].shape
    salt_vs_pepper = 0.2
    amount = 0.004
    num_salt = np.ceil(amount * X_sp[0].size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_sp[0].size * (1.0 - salt_vs_pepper))

    for X_img in X_sp:
        # Add Salt noise
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 1

        # Add Pepper noise
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
        X_img[coords[0], coords[1], :] = 0
    return X_sp, Y_sp


def add_gaussian_noise(X_imgs, Y_imgs):
    X_gn = []
    Y_gn = Y_imgs
    row, col, _ = X_imgs[0].shape
    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5

    for X_img in X_imgs:
        gaussian = np.random.random((row, col, 1)).astype(np.float32)
        gaussian = np.concatenate((gaussian, gaussian, gaussian), axis = 2)
        gaussian_img = cv2.addWeighted(X_img.astype(np.float32), 0.75, 0.25 * gaussian, 0.25, 0)
        X_gn.append(gaussian_img)
    X_gn = np.array(X_gn, dtype = np.float32)
    return X_gn, Y_gn
