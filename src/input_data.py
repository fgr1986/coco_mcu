# MIT License
#
# Copyright (C) 2019 Arm Limited or its affiliates. All rights reserved.
#
# Authors: Fernando García Redondo and Javier Fernández Marqués
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import tensorflow as tf

import argparse
import numpy as np
import json

from tensorflow.python.platform import gfile
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import image_ops


slim = tf.contrib.slim

_VAL_MIN_IDS_FILE = './mscoco_minival_ids.txt'
_FILE_PATTERN = '%s.record-*'

_NUM_CLASSES = 2
IMG_SIZE = 256

_SPLITS_TO_SIZES = {
    'trn': 82783,
    'val': 40504,
}

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, an integer in {0, 1}',
    'object/bbox': 'A list of bounding boxes.',
    'object/label': 'A list of labels, all objects belong to the same class.',
}


parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ("True", "true", "1")


_TF_RECORDS_FOLDER = 'path/to/processed/dataset'
print('_TF_RECORDS_FOLDER is harcoded')

# labels file
LABELS_FILENAME = 'labels.txt'
INPUT_IMAGE_SIZE = [IMG_SIZE, IMG_SIZE, 3]

TRN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 50

features = {
    'image/encoded':
        tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format':
        tf.FixedLenFeature((), tf.string, default_value='jpeg'),
    'image/class/label':
        tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
    'image/object/bbox/xmin':
        tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymin':
        tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/xmax':
        tf.VarLenFeature(dtype=tf.float32),
    'image/object/bbox/ymax':
        tf.VarLenFeature(dtype=tf.float32),
    'image/object/class/label':
        tf.VarLenFeature(dtype=tf.int64),
}


def decode(image_buffer, image_format):

    def decode_image():
        return tf.cast(image_ops.decode_image(image_buffer, channels=3),
                       tf.uint8)

    def decode_jpeg():
        return tf.cast(image_ops.decode_jpeg(image_buffer, channels=3),
                       tf.uint8)

    def check_jpeg():
        return tf.cond(image_ops.is_jpeg(image_buffer), decode_jpeg,
                       decode_image)

    def decode_raw():
        return parsing_ops.decode_raw(image_buffer, out_type=tf.uint8)

    image = tf.cond(math_ops.logical_or(
        math_ops.equal(image_format, 'raw'),
        math_ops.equal(image_format, 'RAW')),
        decode_raw, check_jpeg)

    image.set_shape([None, None, 3])

    return image


def _parse_function_one_hot_val(example_proto):

    parsed_features = tf.parse_single_example(example_proto, features)
    image_decoded = decode(
        parsed_features["image/encoded"], parsed_features["image/format"])

    image_resized = tf.image.resize_images(image_decoded,
                                           [INPUT_IMAGE_SIZE[0],
                                            INPUT_IMAGE_SIZE[1]])
    label = tf.cast(parsed_features['image/class/label'],
                    dtype=tf.int32)
    # label = tf.keras.utils.to_categorical(label, _NUM_CLASSES)
    label = tf.cast(tf.keras.backend.one_hot(label,
                                             _NUM_CLASSES), dtype=tf.float32)

    return image_resized, label


def _parse_function_one_hot_trn(example_proto):

    parsed_features = tf.parse_single_example(example_proto, features)
    image_decoded = decode(
        parsed_features["image/encoded"], parsed_features["image/format"])

    image_resized = tf.image.resize_images(image_decoded,
                                           [int(1.25*INPUT_IMAGE_SIZE[0]),
                                            int(1.25*INPUT_IMAGE_SIZE[1])])
    label = tf.cast(parsed_features['image/class/label'],
                    dtype=tf.int32)
    # label = tf.keras.utils.to_categorical(label, _NUM_CLASSES)
    label = tf.cast(tf.keras.backend.one_hot(label,
                                             _NUM_CLASSES), dtype=tf.float32)

    return image_resized, label


def _parse_function_val(example_proto):

    parsed_features = tf.parse_single_example(example_proto, features)
    image_decoded = decode(
        parsed_features["image/encoded"], parsed_features["image/format"])

    image_resized = tf.image.resize_images(image_decoded,
                                           [int(INPUT_IMAGE_SIZE[0]),
                                            int(INPUT_IMAGE_SIZE[1])])
    label = tf.cast(parsed_features['image/class/label'],
                    dtype=tf.int32)

    return image_resized, label


def _parse_function_trn(example_proto):

    parsed_features = tf.parse_single_example(example_proto, features)
    image_decoded = decode(
        parsed_features["image/encoded"], parsed_features["image/format"])

    image_resized = tf.image.resize_images(image_decoded,
                                           [int(1.25*INPUT_IMAGE_SIZE[0]),
                                            int(1.25*INPUT_IMAGE_SIZE[1])])
    label = tf.cast(parsed_features['image/class/label'],
                    dtype=tf.int32)

    return image_resized, label


def _data_normalization(image, label):
    image = tf.image.resize_images(image,
                                   [INPUT_IMAGE_SIZE[0],
                                    INPUT_IMAGE_SIZE[1]])

    # image = tf.clip_by_value(image, 0, 255.)/255.
    # after what it is written in the challenge description
    image = tf.cast(tf.clip_by_value(
        tf.cast(image, tf.int32),
        0, 255), tf.float32)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    mean = tf.constant(IMAGENET_MEAN, tf.float32)*255.0
    var = tf.constant(IMAGENET_STD, tf.float32)*255.0
    image = (image - mean)/var
    return image, label


def _data_augmentation(image, label):
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.8, 1.1)
    image = tf.image.random_hue(image, 0.1)
    # image = tf.image.random_jpeg_quality(image, 90, 100)
    image = tf.image.random_crop(image,
                                 [INPUT_IMAGE_SIZE[0],
                                  INPUT_IMAGE_SIZE[1],
                                  3])
    # tf.image.random_flip_left_right
    # tf.image.random_brightness
    # tf.image.random_hue
    # tf.image.random_jpeg_quality
    return image, label


def _get_file_names(file_pattern, randomize_input):

    if isinstance(file_pattern, list):
        if not file_pattern:
            raise ValueError('No files given to dequeue_examples.')
        file_names = []
        for entry in file_pattern:
            file_names.extend(gfile.Glob(entry))
    else:
        file_names = list(gfile.Glob(file_pattern))

    if not file_names:
        raise ValueError('No files match %s.' % file_pattern)

    # Sort files so it will be deterministic for unit tests.
    # They'll be shuffled
    # in `string_input_producer` if `randomize_input` is enabled.
    if not randomize_input:
        file_names = sorted(file_names)

    return file_names


def get_dataset(train=True,
                do_one_hot=True,
                repeat_epochs=1):

    if train:
        split_name = 'train'
    else:
        # split_name = 'validation'
        split_name = 'val'

    FILE_PATTERN = split_name + '.record*'

    file_pattern = os.path.join(_TF_RECORDS_FOLDER, FILE_PATTERN)

    file_names = _get_file_names(file_pattern, randomize_input=True)

    dataset = tf.data.TFRecordDataset(file_names)

    # select images fn
    if do_one_hot:
        if train:
            read_img_fn = _parse_function_one_hot_trn
        else:
            read_img_fn = _parse_function_one_hot_val
    else:
        if train:
            read_img_fn = _parse_function_trn
        else:
            read_img_fn = _parse_function_val
    # read images
    dataset = dataset.map(read_img_fn, num_parallel_calls=4)

    # data augmentation
    if train:
        dataset = dataset.map(_data_augmentation)
    # data normalization
    dataset = dataset.map(_data_normalization)
    if train:
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.repeat(repeat_epochs).batch(
            TRN_BATCH_SIZE).prefetch(4)
        print('batch_size trn: ', TRN_BATCH_SIZE)
    else:
        dataset = dataset.repeat(repeat_epochs).batch(
            VAL_BATCH_SIZE).prefetch(2)
        print('batch_size val: ', VAL_BATCH_SIZE)
    print('dataset: ', dataset)

    return dataset


def _get_minival_name(id, folder):
    return folder + '/COCO_val_2014_' + id + '.jpg'


def get_report_dataset(minival_ids_path,
                       minival_json_path,
                       minival_img_folder_path,
                       do_one_hot=True,
                       repeat_epochs=1):

    # aux function
    def _parse_minival(filename):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)

        return tf.image.resize_images(image_decoded,
                                      [int(INPUT_IMAGE_SIZE[0]),
                                       int(INPUT_IMAGE_SIZE[1])])

    # ids
    ids = np.genfromtxt(minival_ids_path)
    # json
    with open(minival_json_path, "r") as f:
        json_data = json.load(f)
    labels = [None] * ids.shape[0]
    file_names = [None] * ids.shape[0]

    all_annotations = json_data['annotations']
    all_images = json_data['images']
    found_idx = 0
    for tmp_image in all_images:
        image_id = int(tmp_image['id'])
        if image_id not in ids:
            continue
        print('found: ', image_id)
        tmp_an = all_annotations[str(image_id)][0]
        file_names[found_idx] = os.path.join(minival_img_folder_path,
                                             tmp_image['file_name'])
        labels[found_idx] = tmp_an['label']
        found_idx += 1

    files_dataset = tf.data.Dataset.from_tensor_slices(file_names)
    labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
    images_dataset = files_dataset.map(_parse_minival)

    dataset = tf.data.Dataset.zip((images_dataset, labels_dataset))

    # data normalization
    dataset = dataset.map(_data_normalization)
    dataset = dataset.repeat(repeat_epochs).batch(
        VAL_BATCH_SIZE).prefetch(2)
    print('batch_size val: ', VAL_BATCH_SIZE)
    print('dataset: ', dataset)

    return dataset
