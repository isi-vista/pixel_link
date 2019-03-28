#!/bin/env python
#encoding = utf-8

import os
import numpy as np
import math
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.training.python.training import evaluation
from datasets import dataset_factory
from preprocessing import ssd_vgg_preprocessing
from tf_extended import metrics as tfe_metrics
import util
import cv2
import pixel_link
from nets import pixel_link_symbol


slim = tf.contrib.slim
import config
# =========================================================================== #
# Checkpoint and running Flags
# =========================================================================== #
tf.app.flags.DEFINE_string('checkpoint_path', None, 
   'the path of pretrained model to be used. If there are checkpoints\
    in train_dir, this config will be ignored.')

tf.app.flags.DEFINE_string('output_dir', None, 
   'the directory for output bounding boxes.')

tf.app.flags.DEFINE_float('gpu_memory_fraction', -1, 
  'the gpu memory fraction to be used. If less than 0, allow_growth = True is used.')


# =========================================================================== #
# I/O and preprocessing Flags.
# =========================================================================== #
tf.app.flags.DEFINE_integer(
    'num_readers', 1,
    'The number of parallel readers that read data from the dataset.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_bool('preprocessing_use_rotation', False, 
             'Whether to use rotation for data augmentation')

# =========================================================================== #
# Dataset Flags.
# =========================================================================== #
tf.app.flags.DEFINE_string('dataset_dir', 
           util.io.get_absolute_path('~/dataset/ICDAR2015/Challenge4/ch4_test_images'), 
           'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer('eval_image_width', 1280, 'Train image size')

tf.app.flags.DEFINE_integer('eval_image_height', 768, 'Train image size')

tf.app.flags.DEFINE_bool('using_moving_average', True, 
                         'Whether to use ExponentionalMovingAverage')

tf.app.flags.DEFINE_float('moving_average_decay', 0.9999, 
                          'The decay rate of ExponentionalMovingAverage')


FLAGS = tf.app.flags.FLAGS

def config_initialization():
    # image shape and feature layers shape inference
    image_shape = (FLAGS.eval_image_height, FLAGS.eval_image_width)
    
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')
    
    tf.logging.set_verbosity(tf.logging.DEBUG)
    config.load_config(FLAGS.checkpoint_path)
    config.load_config(FLAGS.output_dir)
    config.init_config(image_shape, 
                       batch_size = 1, 
                       pixel_conf_threshold = 0.8,
                       link_conf_threshold = 0.8,
                       num_gpus = 1, 
                   )

def to_txt(txt_path, image_name, 
           image_data, pixel_pos_scores, link_pos_scores):
    # write detection result as txt files
    def write_result_as_txt(image_name, bboxes, path):
        filename = util.io.join_path(path, '%s.txt'%(image_name))
        lines = []
        for b_idx, bbox in enumerate(bboxes):
              values = [int(v) for v in bbox]
              line = "%d, %d, %d, %d, %d, %d, %d, %d\n"%tuple(values)
              lines.append(line)
        util.io.write_lines(filename, lines)
        print('result has been written to:', filename)
    
    mask = pixel_link.decode_batch(pixel_pos_scores, link_pos_scores)[0, ...]
    bboxes = pixel_link.mask_to_bboxes(mask, image_data.shape)
    lines = pixel_link.bboxes_to_lines(bboxes)
    write_result_as_txt(image_name, lines, txt_path)

def test():
    with tf.name_scope('test'):
        image = tf.placeholder(dtype=tf.int32, shape = [None, None, 3])
        image_shape = tf.placeholder(dtype = tf.int32, shape = [3, ])
        processed_image, _, _, _, _ = ssd_vgg_preprocessing.preprocess_image(image, None, None, None, None, 
                                                   out_shape = config.image_shape,
                                                   data_format = config.data_format, 
                                                   is_training = False)
        b_image = tf.expand_dims(processed_image, axis = 0)
        net = pixel_link_symbol.PixelLinkNet(b_image, is_training = True)
        global_step = slim.get_or_create_global_step()

    
    sess_config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True)
    if FLAGS.gpu_memory_fraction < 0:
        sess_config.gpu_options.allow_growth = True
    elif FLAGS.gpu_memory_fraction > 0:
        sess_config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction;
    

    # Variables to restore: moving avg. or normal weights.
    if FLAGS.using_moving_average:
        variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay)
        variables_to_restore = variable_averages.variables_to_restore()
        variables_to_restore[global_step.op.name] = global_step
    else:
        variables_to_restore = slim.get_variables_to_restore()
    
    saver = tf.train.Saver(var_list = variables_to_restore)

    video_names = util.io.ls(FLAGS.dataset_dir)
    video_names.sort()
    
    checkpoint = FLAGS.checkpoint_path
    checkpoint_dir = util.io.get_dir(FLAGS.checkpoint_path)
    output_dir = util.io.get_dir(FLAGS.output_dir)
    
    with tf.Session(config = sess_config) as sess:
        saver.restore(sess, checkpoint)

        for iter, video_name in enumerate(video_names):
            basename = os.path.splitext(os.path.basename(video_name))[0]
            vidcap = cv2.VideoCapture(util.io.join_path(FLAGS.dataset_dir,video_name))
            length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = 100

            success,image_data = vidcap.read()
            count = 1
            while success:
              if count % step != 0:
                success,image_data = vidcap.read()
                count += 1
                continue

              pixel_pos_scores, link_pos_scores = sess.run(
                  [net.pixel_pos_scores, net.link_pos_scores], 
                  feed_dict = { image:image_data })

              image_name = basename+'-'+str(count)+'.jpg'

              to_txt(output_dir,
                      image_name, image_data, 
                      pixel_pos_scores, link_pos_scores)

              success,image_data = vidcap.read()
              count += 1

            print('%d/%d: %s'%(iter + 1, len(video_names), video_name))

def main(_):
    config_initialization()
    test()
    
if __name__ == '__main__':
    tf.app.run()
