import argparse
import logging
import os
import cv2
import numpy as np
import tensorflow as tf
import json

import matplotlib.pyplot as plt

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def model_wh(resolution_str):
    width, height = map(int, resolution_str.split('x'))
    if width % 16 != 0 or height % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(width), int(height)


def read_img(path, width, height):
    val_image = cv2.imread(path, cv2.IMREAD_COLOR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def graph_pred(graph, test_images, resize):
    # test_images = []
    # graph tf.graph
    # resize tuple

    # from tensor
    x = graph.get_tensor_by_name('prefix/image:0')
    # to tensor
    y = graph.get_tensor_by_name('prefix/Openpose/concat_stage7:0')

    def feature(test_features):
        return cv2.resize(test_features, resize)[np.newaxis, :, :, :].astype(np.float32)

    pred_y = []
    with tf.Session(graph=graph) as sess:
        for test_image in test_images:
            pred_y_tmp = sess.run(y, feed_dict={x: feature(test_image)})
            pred_y.append(pred_y_tmp)

    return pred_y


# TODO: ckpt loader not support

def keypoints_gen(pred_y, image_size, need_cols):
    # image_size
    im_with_keypoints = {}
    im_with_position = {}
    for k, col in enumerate(need_cols):
        pred_y_tmp = 255 - cv2.resize(pred_y[0], image_size[::-1])[:, :, k:k + 1]

        im = pred_y_tmp.copy()

        def float2unit8(im):
            return (255 * (im - np.min(im)) / (np.max(im) - np.min(im))).astype(np.uint8)

        im = float2unit8(im)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        params = cv2.SimpleBlobDetector_Params()

        # # Change thresholds
        # params.minThreshold = 10
        # params.maxThreshold = 200
        #
        # # Filter by Area.
        # params.filterByArea = True
        # params.minArea = 15
        #
        # # Filter by Circularity
        # params.filterByCircularity = True
        # params.minCircularity = 0.1
        #
        # # Filter by Convexity
        # params.filterByConvexity = True
        # params.minConvexity = 0.87
        #
        # # Filter by Inertia
        # params.filterByInertia = True
        # params.minInertiaRatio = 0.01

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(im)
        positions = [n.pt for n in keypoints]

        im_with_keypoints_tmp = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                                  cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        im_with_keypoints[col] = im_with_keypoints_tmp
        im_with_position[col] = positions

    return {"keypoint_detect": im_with_keypoints, "keypoint_position": im_with_position}


def compare_default(key_points_data):
    return key_points_data["keypoint_position"]


def compare_unique(key_points_data):
    # input: keypoint_detect: <col, np.array>, keypoint_position: <col, list>
    # "neckline_left", "neckline_right",
    # "center_front",
    # "shoulder_left", "shoulder_right",
    # "armpit_left", "armpit_right",
    # "cuff_left_in", "cuff_right_in",
    # "cuff_left_out","cuff_right_out",
    # "top_hem_left", "top_hem_right"

    def pair_pos(a, b):
        # a : [pos1, pos2]
        # b : [pos1, pos2]
        # left, right
        # only compare pairs less than 3
        if len(a) * len(b) == 1:
            tmp = sorted(a + b)
            return tmp[:1], tmp[-1:]
        elif len(a) * len(b) == 2:
            tmp = (a if len(a) == 2 else []) + (b if len(b) == 2 else [])
            return tmp[:1], tmp[-1:]
        elif len(a) == 2 and len(b) == 2:
            tmp = sorted(a)[:1] + sorted(b)[-1:]
            return tmp[:1], tmp[-1:]
        else:
            return a, b

    x = key_points_data["keypoint_position"]

    x["neckline_left"], x["neckline_right"] = pair_pos(x["neckline_left"], x["neckline_right"])
    x["shoulder_left"], x["shoulder_right"] = pair_pos(x["shoulder_left"], x["shoulder_right"])
    x["armpit_left"], x["armpit_right"] = pair_pos(x["armpit_left"], x["armpit_right"])
    x["cuff_left_in"], x["cuff_right_in"] = pair_pos(x["cuff_left_in"], x["cuff_right_in"])
    x["cuff_left_out"], x["cuff_right_out"] = pair_pos(x["cuff_left_out"], x["cuff_right_out"])
    x["top_hem_left"], x["top_hem_right"] = pair_pos(x["top_hem_left"], x["top_hem_right"])

    return x


def draw_keypoint(key_points_data, im_origin, save_path):
    im_with_keypoints = key_points_data["keypoint_detect"]
    im_with_position = key_points_data["keypoint_position"]

    fig = plt.figure(figsize=(20, 20))
    for k, (col, im) in enumerate(im_with_keypoints.items()):
        fig.add_subplot(4, 4, k + 1)
        # im and keypoint dectect in one picture
        im_plus_keypoint = 255 - im + cv2.cvtColor(im_origin, cv2.COLOR_BGR2RGB)
        plt.imshow(im_plus_keypoint / np.max(im_plus_keypoint, axis=(0, 1)))

        # a = 255 - im
        # a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        # b = im_origin.copy()
        # th = max(1, len(im_with_position[col]))
        # b[:, :, 0] = b[:, :, 0] * 0.3 + a * 0.7 / th
        # b[:, :, 1] = b[:, :, 1] * 0.3
        # b[:, :, 2] = b[:, :, 2] * 0.3
        # im_plus_keypoint = b
        # plt.imshow(cv2.cvtColor(im_plus_keypoint, cv2.COLOR_BGR2RGB))
        plt.title(col + " %s" % len(im_with_position[col]))
    fig.savefig(save_path)
    plt.close()
    return save_path


parser = argparse.ArgumentParser(description='tf-pose-estimation run')
parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin/ model_path')
parser.add_argument('--testdir', type=str, default='[None]', help='test dir')
parser.add_argument('--outputdir', type=str, default='[None]', help='output dir')
parser.add_argument('--traindir', type=str, default='[None]', help='train dir')
HOME_PATH = "/media/yanpan/7D4CF1590195F939"
# HOME_PATH = "D:"
args = parser.parse_args(
    f"--model {HOME_PATH}/Projects/tf-pose-model/myblouse/tf-pose-2-blouse/graph_freeze.pb "
    "--resolution 368x368 "
    f"--traindir {HOME_PATH}/Projects/fashionai/mytrain/myblouse "
    f"--outputdir {HOME_PATH}/Projects/fashionai/pred/tf-pose-2-blouse/blouse "
    f"--testdir {HOME_PATH}/Projects/fashionai/test/Images/blouse".split())

image_paths = [args.testdir + "/" + x for x in os.listdir(args.testdir)]
images = [read_img(image_path, None, None) for image_path in image_paths]
image_sizes = [test_image.shape[:2] for test_image in images]
with open(args.traindir + "/annotations/need_cols.txt", "r", encoding="utf8") as f:
    need_cols = [x.strip() for x in f.readlines()]

if __name__ == '__test__':
    w, h = model_wh(args.resolution)
    graph = load_graph(args.model)
    pred_images = graph_pred(graph, images, (368, 368))
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    with open(args.outputdir + "/" + "pred_images.npy", "wb") as f:
        np.save(f, pred_images)

if __name__ == '__main__':
    with open(args.outputdir + "/" + "pred_images.npy", "rb") as f:
        pred_images = np.load(f)

    # pred_images_dict is a list of dicts, every dict contains
    outcome = []
    for k, pred_image in enumerate(pred_images):

        pred_image_im = keypoints_gen(pred_image, image_sizes[k], need_cols[2:])
        # compare_unique: generate pred_image_keypoint_position, single per keypoint
        # pred_image_keypoint_position = compare_unique(pred_image_im)
        pred_image_keypoint_position = compare_default(pred_image_im)

        # detect if pred_image_keypoint_position has more than 1 points
        count = {}
        for col, pos in pred_image_keypoint_position.items():
            if len(pos) > 1:
                count[col] = len(pos)
        if len(count) > 0:
            print("more than 1 points! %5s" % k, image_paths[k], count)
            #
            # if k % 10000 == 10000 - 1:
            save_path = args.outputdir + "/" + image_paths[k].split("/")[-1]
            draw_keypoint(pred_image_im, images[k], save_path)

        outcome_dict_tmp = {"id": image_paths[k], "pos": pred_image_keypoint_position}
        outcome.append(outcome_dict_tmp)

    with open(args.outputdir + "/pred.json", "w") as f:
        json.dump(outcome, f)

if __name__ == '__tmp__':
    ppp = f"{HOME_PATH}/Projects/fashionai/test/Images/blouse/%s.jpg"
    ppp = ppp % "0a77868d2697cfa21f400e4cedc972c9"
    k = image_paths.index(ppp)
    pred_image = pred_images[k]

# 重叠两个区域，选取覆盖最多的那个像素块：2值化最黑的地方
# 然后遇到覆盖最多的超过2个，就删除已经被前面找到的那些位置
