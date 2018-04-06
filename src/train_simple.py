import matplotlib as mpl

mpl.use('Agg')  # training mode, no screen should be open. (It will block training loop)

import argparse
import logging
import os
import time
import re

os.environ["OPENPOSE_MODEL"] = "/media/yanpan/7D4CF1590195F939/Projects/tf-pose-es/models"

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorpack.dataflow.remote import RemoteDataZMQ

from pose_dataset import get_dataflow_batch, DataFlowToQueue, CocoPose
from pose_augment import set_network_input_wh, set_network_scale
from common import get_sample_images
from networks import get_network

logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

from taskdesc import TASK_KEY_POINTS

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
    parser.add_argument('--model', default='mobilenet', help='model name')
    parser.add_argument('--datapath', type=str, default='/root/coco/annotations')
    parser.add_argument('--imgpath', type=str, default='/root/coco/')
    parser.add_argument('--batchsize', type=int, default=96)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--max-epoch', type=int, default=30)
    parser.add_argument('--lr', type=str, default='0.01')
    parser.add_argument('--modelpath', type=str, default='/data/private/tf-openpose-models-2018-1/')
    parser.add_argument('--ckptpath', type=str, default='')
    parser.add_argument('--logpath', type=str, default='/data/private/tf-openpose-log-2018-1/')
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--remote-data', type=str, default='', help='eg. tcp://0.0.0.0:1027')

    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--sample_path', help="sample_path to eval during training, limit 12")
    args = parser.parse_args("""--model cmu
                                --datapath /media/yanpan/7D4CF1590195F939/Projects/fashionai/mytrain/myblouse/annotations/ 
                                --imgpath /media/yanpan/7D4CF1590195F939/Projects/fashionai/mytrain/myblouse/
                                --sample_path /media/yanpan/7D4CF1590195F939/Projects/fashionai/mytrain/myblouse/val2017
                                --batchsize 32 --lr 0.0001 
                                --modelpath /media/yanpan/7D4CF1590195F939/Projects/tf-pose-model/myblouse/tf-pose-3-blouse/ 
                                --logpath /media/yanpan/7D4CF1590195F939/Projects/tf-pose-model/myblouse/tf-pose-3-blouse/
                                """.split())
    # --ckptpath /media/yanpan/7D4CF1590195F939/Projects/mytrain/myblouse/tf-pose-1-blouse/cmu_batch:32_lr:0.0001_gpus:1_368x368_/checkpoint
    # --ckptpath /media/yanpan/7D4CF1590195F939/Projects/tf-pose-5/cmu_batch:32_lr:0.001_gpus:1_368x368_/checkpoint

    if args.gpus <= 0:
        raise Exception('gpus <= 0')

    # define input placeholder
    set_network_input_wh(args.input_width, args.input_height)
    scale = 4

    if args.model in ['cmu', 'vgg', 'mobilenet_thin', 'mobilenet_try', 'mobilenet_try2', 'mobilenet_try3',
                      'hybridnet_try']:
        scale = 8

    set_network_scale(scale)
    output_w, output_h = args.input_width // scale, args.input_height // scale

    logger.info('define model+')
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=0)):
        input_node = tf.placeholder(tf.float32, shape=(args.batchsize, args.input_height, args.input_width, 3),
                                    name='image')
        vectmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, TASK_KEY_POINTS * 2),
                                      name='vectmap')
        heatmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, TASK_KEY_POINTS),
                                      name='heatmap')

        # prepare data
        if not args.remote_data:
            df = get_dataflow_batch(args.datapath, True, args.batchsize, img_path=args.imgpath, if_filp=False)
        else:
            # transfer inputs from ZMQ
            df = RemoteDataZMQ(args.remote_data, hwm=3)
        enqueuer = DataFlowToQueue(df, [input_node, heatmap_node, vectmap_node], queue_size=100)
        q_inp, q_heat, q_vect = enqueuer.dequeue()

    df_valid = get_dataflow_batch(args.datapath, False, args.batchsize, img_path=args.imgpath, buffer_size=500,
                                  if_filp=False)
    # TODO: auto dectect buffer_size
    df_valid.reset_state()
    validation_cache = []

    val_image = get_sample_images(args.sample_path, args.input_width, args.input_height)
    VALID_SIZE = len(val_image)
    logger.info('tensorboard val image: %d' % len(val_image))
    logger.info(q_inp)
    logger.info(q_heat)
    logger.info(q_vect)

    # define model for multi-gpu
    q_inp_split, q_heat_split, q_vect_split = tf.split(q_inp, args.gpus), tf.split(q_heat, args.gpus), tf.split(q_vect,
                                                                                                                args.gpus)

    # output_vectmap = []
    output_heatmap = []
    # losses = []
    # last_losses_l1 = []
    last_losses_l2 = []
    outputs = []
    for gpu_id in range(args.gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                net, pretrain_path, last_layer = get_network(args.model, q_inp_split[gpu_id])
                # TODO:
                if len(args.ckptpath) > 1:
                    with open(args.ckptpath, "r", encoding="utf8") as f:
                        ckpt = f.readlines()
                    ckptfind = re.search("\"(.*)\"", ckpt[0][:-1])
                    pretrain_path = ckptfind.group(1)
                else:
                    pretrain_path = None

                heat = net.loss_last()  # vect,
                # output_vectmap.append(vect)
                output_heatmap.append(heat)
                outputs.append(net.get_output())

                l2s = net.loss_l1_l2()  # l1s,
                for idx, l2 in enumerate(l2s):  # for idx, (l1, l2) in enumerate(zip(l1s, l2s)):
                    # loss_l1 = tf.nn.l2_loss(tf.concat(l1, axis=0) - q_vect_split[gpu_id],
                    #                         name='loss_l1_stage%d_tower%d' % (idx, gpu_id))
                    loss_l2 = tf.nn.l2_loss(tf.concat(l2, axis=0) - q_heat_split[gpu_id],
                                            name='loss_l2_stage%d_tower%d' % (idx, gpu_id))
                    # losses.append(tf.reduce_mean([loss_l2]))  # loss_l1,

                # last_losses_l1.append(loss_l1)
                last_losses_l2.append(loss_l2)

    outputs = tf.concat(outputs, axis=0)

    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
        # define loss
        # total_loss = tf.reduce_sum(losses) / args.batchsize
        # total_loss_ll_paf = tf.reduce_sum(last_losses_l1) / args.batchsize
        # total_loss_ll = tf.reduce_mean([total_loss_ll_paf, total_loss_ll_heat])
        total_loss_ll_heat = tf.reduce_sum(last_losses_l2) / args.batchsize
        # total_loss_ll = tf.reduce_mean([total_loss_ll_heat])

        # define optimizer
        step_per_epoch = 121745 // args.batchsize
        global_step = tf.Variable(0, trainable=False)
        if ',' not in args.lr:
            starter_learning_rate = float(args.lr)
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       decay_steps=5000, decay_rate=0.5, staircase=True)
            # decay_steps=10000, decay_rate=0.33, staircase=True)
        else:
            lrs = [float(x) for x in args.lr.split(',')]
            boundaries = [step_per_epoch * 5 * i for i, _ in range(len(lrs)) if i > 0]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, lrs)

    # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.0005, momentum=0.9, epsilon=1e-10)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss_ll_heat, global_step, colocate_gradients_with_ops=True)
    logger.info('define model-')

    # define summary
    # tf.summary.scalar("loss", total_loss)
    # tf.summary.scalar("loss_lastlayer", total_loss_ll)
    # tf.summary.scalar("loss_lastlayer_paf", total_loss_ll_paf)
    tf.summary.scalar("loss_lastlayer_heat", total_loss_ll_heat)
    tf.summary.scalar("queue_size", enqueuer.size())
    tf.summary.scalar("learning_rate", optimizer._lr)
    merged_summary_op = tf.summary.merge_all()

    valid_loss = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll_paf = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll_heat = tf.placeholder(tf.float32, shape=[])
    sample_train = tf.placeholder(tf.float32, shape=(4, 640, 640, 3))
    sample_valid = tf.placeholder(tf.float32, shape=(12, 640, 640, 3))
    train_img = tf.summary.image('training sample', sample_train, 4)
    valid_img = tf.summary.image('validation sample', sample_valid, 12)
    valid_loss_t = tf.summary.scalar("loss_valid", valid_loss)
    valid_loss_ll_t = tf.summary.scalar("loss_valid_lastlayer", valid_loss_ll)
    merged_validate_op = tf.summary.merge([train_img, valid_img, valid_loss_t, valid_loss_ll_t])

    saver = tf.train.Saver(max_to_keep=100)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # config.gpu_options.per_process_gpu_memory_fraction = 0.95  # 程序最多只能占用指定gpu50%的显存
    # config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        training_name = '{}_batch:{}_lr:{}_gpus:{}_{}x{}_{}'.format(
            args.model,
            args.batchsize,
            args.lr,
            args.gpus,
            args.input_width, args.input_height,
            args.tag
        )
        logger.info('model weights initialization')
        sess.run(tf.global_variables_initializer())

        if args.checkpoint:
            logger.info('Restore from checkpoint...')
            # loader = tf.train.Saver(net.restorable_variables())
            # loader.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            saver.restore(sess, tf.train.latest_checkpoint(args.checkpoint))
            logger.info('Restore from checkpoint...Done')
        elif pretrain_path:
            logger.info('Restore pretrained weights...')
            # if '.ckpt' in pretrain_path:
            if 'model-' in pretrain_path:
                loader = tf.train.Saver(net.restorable_variables())
                loader.restore(sess, pretrain_path)
            elif '.npy' in pretrain_path:
                net.load(pretrain_path, sess, False)
            logger.info('Restore pretrained weights...Done')

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

        while True:
            _, gs_num = sess.run([train_op, global_step])

            if gs_num > step_per_epoch * args.max_epoch:
                break

            if gs_num - last_gs_num >= 100:
                train_loss_ll_heat, lr_val, summary, queue_size = sess.run(
                    # train_loss_ll_paf
                    [total_loss_ll_heat, learning_rate, merged_summary_op,
                     # total_loss, total_loss_ll,  total_loss_ll_paf
                     enqueuer.size()])

                # log of training loss / accuracy
                batch_per_sec = (gs_num - initial_gs_num) / (time.time() - time_started)
                logger.info(
                    'epoch=%.2f step=%d, %0.4f examples/sec lr=%f, loss_ll_heat=%g, q=%d' % (  # loss=%g, loss_ll=%g,
                        # loss_ll_paf=%g,
                        gs_num / step_per_epoch, gs_num, batch_per_sec * args.batchsize, lr_val,
                        # train_loss, train_loss_ll,
                        train_loss_ll_heat, queue_size))  # train_loss_ll_paf
                last_gs_num = gs_num

                file_writer.add_summary(summary, gs_num)

            if gs_num - last_gs_num2 >= 1000:
                # save weights
                saver.save(sess, os.path.join(args.modelpath, training_name, 'model'), global_step=global_step)

                average_loss = average_loss_ll = average_loss_ll_paf = average_loss_ll_heat = 0
                total_cnt = 0

                if len(validation_cache) == 0:
                    for images_test, heatmaps, vectmaps in tqdm(df_valid.get_data()):
                        validation_cache.append((images_test, heatmaps, vectmaps))
                    df_valid.reset_state()
                    del df_valid
                    df_valid = None

                # log of test accuracy
                for images_test, heatmaps, vectmaps in validation_cache:
                    lss_ll_heat = sess.run(  # lss_ll_paf, vectmap_sample lss, lss_ll,, heatmap_sample
                        total_loss_ll_heat,
                        # total_loss, total_loss_ll, output_vectmap,  # total_loss_ll_paf,output_heatmap
                        feed_dict={q_inp: images_test, q_vect: vectmaps, q_heat: heatmaps}
                    )
                    # average_loss += lss * len(images_test)
                    # average_loss_ll += lss_ll * len(images_test)
                    # average_loss_ll_paf += lss_ll_paf * len(images_test)
                    average_loss_ll_heat += lss_ll_heat * len(images_test)
                    total_cnt += len(images_test)

                logger.info('validation(%d) %s , loss_ll_heat=%f' % (  # loss_ll_paf=%f,loss=%f, loss_ll=%f
                    total_cnt, training_name,  # average_loss / total_cnt, average_loss_ll / total_cnt,
                    average_loss_ll_heat / total_cnt))  # average_loss_ll_paf / total_cnt,
                last_gs_num2 = gs_num

                sample_image = [enqueuer.last_dp[0][i] for i in range(4)]
                outputMat = sess.run(
                    outputs,
                    feed_dict={q_inp: np.array((sample_image + val_image) * (args.batchsize // 16))}
                )
                pafMat, heatMat = outputMat[:, :, :, 4:], outputMat[:, :, :, :4]

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
                    valid_loss: average_loss / total_cnt,
                    valid_loss_ll: average_loss_ll / total_cnt,
                    # valid_loss_ll_paf: average_loss_ll_paf / total_cnt,
                    valid_loss_ll_heat: average_loss_ll_heat / total_cnt,
                    sample_valid: test_results,
                    sample_train: sample_results
                })
                file_writer.add_summary(summary, gs_num)

        saver.save(sess, os.path.join(args.modelpath, training_name, 'model'), global_step=global_step)
    logger.info('optimization finished. %f' % (time.time() - time_started))
