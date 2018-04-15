import matplotlib as mpl

mpl.use('Agg')  # training mode, no screen should be open. (It will block training loop)

import argparse
import logging
import os
import time
import re

import sys

sys.path.append("/media/yanpan/7D4CF1590195F939/Projects/tf-pose-es/src")
os.environ["OPENPOSE_MODEL"] = "/media/yanpan/7D4CF1590195F939/Projects/tf-pose-es/models"

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from pose_dataset import get_dataflow_batch, DataFlowToQueue, CocoPose
from pose_augment import set_network_input_wh, set_network_scale
from common import get_sample_images
from networks import get_network

from taskdesc import TASK_KEY_POINTS as KP

logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def max_l2_dist(a, b, name=None):
    # a0 = tf.argmax(tf.reshape(a, (a.shape[0], a.shape[1] * a.shape[2], a.shape[3])), 1)
    # b0 = tf.argmax(tf.reshape(b, (b.shape[0], b.shape[1] * b.shape[2], b.shape[3])), 1)
    # s = tf.sqrt(tf.to_float((a0 // a.shape[2] - b0 // b.shape[2]) ** 2 + (a0 % a.shape[2] - b0 % b.shape[2]) ** 2))

    return tf.reduce_sum(tf.sqrt(tf.to_float(
        (tf.argmax(tf.reshape(a, (32, 46 * 46, 14))) // 46 -
         tf.argmax(tf.reshape(b, (32, 46 * 46, 14))) // 46) ** 2 +
        (tf.argmax(tf.reshape(a, (32, 46 * 46, 14))) % 46 -
         tf.argmax(tf.reshape(b, (32, 46 * 46, 14))) % 46) ** 2
    )), name=name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='mobilenet', help='model name')
    parser.add_argument('--datapath', type=str, default='/root/coco/annotations')
    parser.add_argument('--imgpath', type=str, default='/root/coco/')
    parser.add_argument('--batchsize', type=int, default=96)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max-epoch', type=int, default=10)
    parser.add_argument('--lr', type=str, default='0.01')
    parser.add_argument('--modelpath', type=str, default='/data/private/tf-openpose-models-2018-1/')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--logpath', type=str, default='/data/private/tf-openpose-log-2018-1/')

    parser.add_argument('--inputwidth', type=int, default=368)
    parser.add_argument('--inputheight', type=int, default=368)
    parser.add_argument('--sample_path', help="sample_path to eval during training, limit 12")

    home_path = "/media/yanpan/7D4CF1590195F939/Projects"
    category = "blouse"
    # 'blouse', 'dress', 'outwear', 'skirt', 'trousers'
    args = parser.parse_args("--model cmu "
                             f"--datapath {home_path}/fashionai/mytrain/my{category}_prof/annotations/ "
                             f"--imgpath {home_path}/fashionai/mytrain/my{category}_prof/ "
                             f"--sample_path {home_path}/fashionai/mytrain/my{category}_prof/val2017 "
                             "--batchsize 32 --lr 0.0001 "
                             f"--modelpath {home_path}/tf-pose-model/my{category}_prof/tf-pose-2-{category}/ "
                             f"--logpath {home_path}/tf-pose-model/my{category}_prof/tf-pose-2-{category}/"
                             "".split())

    # "--checkpoint {home_path}/tf-pose-model/my{category}_prof/tf-pose-1-blouse/cmu_batch:32_lr:0.0001_gpus:1_368x368_"
    # define input placeholder
    set_network_input_wh(args.inputwidth, args.inputheight)
    scale = 8
    set_network_scale(scale)
    output_w, output_h = args.inputwidth // scale, args.inputheight // scale

    logger.info('define model+')
    input_node = tf.placeholder(tf.float32, shape=(args.batchsize, args.inputheight, args.inputwidth, 3), name='image')
    heatmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, KP), name='heatmap')
    # prepare data
    df = get_dataflow_batch(args.datapath, True, args.batchsize, img_path=args.imgpath, if_filp=False)
    enqueuer = DataFlowToQueue(df, [input_node, heatmap_node], queue_size=100)
    q_inp, q_heat = enqueuer.dequeue()

    df_valid = get_dataflow_batch(args.datapath, False, args.batchsize, img_path=args.imgpath, if_filp=False,
                                  buffer_size=500)  # TODO: auto dectect buffer_size
    df_valid.reset_state()
    validation_cache = []

    val_image = get_sample_images(args.sample_path, args.inputwidth, args.inputheight)
    logger.info('tensorboard val image: %d' % len(val_image))
    logger.info(q_inp)
    logger.info(q_heat)

    output_heatmap = []
    last_losses_l2 = []
    outputs = []
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        net, pretrain_path, last_layer = get_network(args.model, q_inp)

        heat = net.loss_last()
        output_heatmap.append(heat)
        outputs.append(net.get_output())

        l2s = net.loss_l1_l2()
        for idx, l2 in enumerate(l2s[:-1]):
            loss_l2 = tf.nn.l2_loss(tf.concat(l2, axis=0) - q_heat, name='loss_l2_stage%d_tower%d' % (idx, 0))
        # change loss, get rid of last channel
        loss_l2 = tf.nn.l2_loss(tf.concat(l2s[-1], axis=0)[:, :, :, :-1] - q_heat[:, :, :, :-1],
                                name='loss_l2_stage%d_tower%d' % (len(l2s) - 1, 0))
        last_losses_l2.append(loss_l2)

    outputs = tf.concat(outputs, axis=0)

    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        # define loss
        total_loss_ll_heat = tf.reduce_sum(last_losses_l2) / args.batchsize

        # define optimizer
        step_per_epoch = 121745 // args.batchsize
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = float(args.lr)
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps=5000, decay_rate=0.5,
                                                   staircase=True)  # TODO: why staircase is True??
        # decay_steps=10000, decay_rate=0.33, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss_ll_heat, global_step, colocate_gradients_with_ops=True)
    logger.info('define model-')

    # define summary
    tf.summary.scalar("loss_lastlayer_heat", total_loss_ll_heat)
    tf.summary.scalar("queue_size", enqueuer.size())
    tf.summary.scalar("learning_rate", optimizer._lr)
    merged_summary_op = tf.summary.merge_all()

    valid_loss_ll_heat = tf.placeholder(tf.float32, shape=[])
    sample_train = tf.placeholder(tf.float32, shape=(4, 640, 640, 3))
    sample_valid = tf.placeholder(tf.float32, shape=(12, 640, 640, 3))
    train_img = tf.summary.image('training_sample', sample_train, 4)
    valid_img = tf.summary.image('validation_sample', sample_valid, 12)
    valid_loss_ll_t = tf.summary.scalar("loss_valid_ll_heat", valid_loss_ll_heat)
    merged_validate_op = tf.summary.merge([train_img, valid_img, valid_loss_ll_t])

    saver = tf.train.Saver(max_to_keep=100)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.95  # 程序最多只能占用指定gpu50%的显存
    # config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        training_name = '{}_batch:{}_lr:{}_gpus:{}_{}x{}_'.format(
            args.model,
            args.batchsize,
            args.lr,
            args.gpus,
            args.inputwidth, args.inputheight,
        )
        logger.info('model weights initialization')
        sess.run(tf.global_variables_initializer())

        if args.checkpoint:
            logger.info('Restore from checkpoint...')
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            logger.info('Restore from checkpoint...Done')

        logger.info('prepare file writer')
        file_writer = tf.summary.FileWriter(args.logpath + training_name, sess.graph)

        logger.info('prepare coordinator')
        coord = tf.train.Coordinator()
        enqueuer.set_coordinator(coord)
        enqueuer.start()

        logger.info('Training Started.')
        time_started = time.time()
        last_gs_num = last_gs_num2 = 0
        initial_gs_num = sess.run(global_step)

        # epoch plot ...
        while True:
            _, gs_num = sess.run([train_op, global_step])

            if gs_num > step_per_epoch * args.max_epoch:
                break

            if gs_num - last_gs_num >= 100:
                l2s_numpy, train_loss_ll_heat, lr_val, summary, queue_size = sess.run(
                    [l2s, total_loss_ll_heat, learning_rate, merged_summary_op, enqueuer.size()])

                # log of training loss / accuracy
                batch_per_sec = (gs_num - initial_gs_num) / (time.time() - time_started)
                logger.info('epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss_ll_heat=%g, q=%d' % (
                    gs_num / step_per_epoch, gs_num, batch_per_sec * args.batchsize, lr_val, train_loss_ll_heat,
                    queue_size))
                last_gs_num = gs_num

                file_writer.add_summary(summary, gs_num)

            if gs_num - last_gs_num2 >= 1000:
                # save weights
                saver.save(sess, os.path.join(args.modelpath, training_name, 'model'), global_step=global_step)

                average_loss_ll_heat = 0
                total_cnt = 0

                if len(validation_cache) == 0:
                    for images_test, heatmaps, vectmaps in tqdm(df_valid.get_data()):
                        validation_cache.append((images_test, heatmaps))
                    df_valid.reset_state()
                    del df_valid
                    df_valid = None

                # log of test accuracy
                for images_test, heatmaps in validation_cache:
                    lss_ll_heat = sess.run(total_loss_ll_heat, feed_dict={q_inp: images_test, q_heat: heatmaps})
                    average_loss_ll_heat += lss_ll_heat * len(images_test)
                    total_cnt += len(images_test)

                logger.info('validation(%d) %s , loss_ll_heat=%f' % (
                    total_cnt, training_name, average_loss_ll_heat / total_cnt))
                last_gs_num2 = gs_num

                sample_image = [enqueuer.last_dp[0][i] for i in range(4)]
                outputMat = sess.run(outputs,
                                     feed_dict={q_inp: np.array((sample_image + val_image) * (args.batchsize // 16))})
                pafMat, heatMat = outputMat[:, :, :, :], outputMat[:, :, :, :]  # pafMat,heatMat useless

                sample_results = []
                for i in range(len(sample_image)):
                    test_result = CocoPose.display_image(sample_image[i], heatMat[i], pafMat[i], as_numpy=True)
                    test_result = cv2.resize(test_result, (640, 640))
                    test_result = test_result.reshape([640, 640, 3]).astype(float)
                    sample_results.append(test_result)

                test_results = []
                for i in range(len(val_image)):
                    test_result = CocoPose.display_image(val_image[i], heatMat[len(sample_image) + i],
                                                         pafMat[len(sample_image) + i], as_numpy=True)
                    test_result = cv2.resize(test_result, (640, 640))
                    test_result = test_result.reshape([640, 640, 3]).astype(float)
                    test_results.append(test_result)

                # save summary
                summary = sess.run(merged_validate_op, feed_dict={
                    valid_loss_ll_heat: average_loss_ll_heat / total_cnt,
                    sample_valid: test_results,
                    sample_train: sample_results
                })
                file_writer.add_summary(summary, gs_num)

        saver.save(sess, os.path.join(args.modelpath, training_name, 'model'), global_step=global_step)
    logger.info('optimization finished. %f' % (time.time() - time_started))
