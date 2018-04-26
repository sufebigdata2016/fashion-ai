import argparse
import logging
import os
import cv2
import numpy as np
import tensorflow as tf
import json
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import re

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

    pred_ys = []
    with tf.Session(graph=graph) as sess:
        for test_image in test_images:
            pred_y_tmp = sess.run(y, feed_dict={x: feature(test_image["image"])})
            test_image["pred_y"] = pred_y_tmp
            pred_ys.append(test_image)
    return pred_ys


def im_pad(im):
    return np.pad(im, [[10, 10], [10, 10], [0, 0]], "maximum")


def im_no_pad(im):
    return im[10:-10, 10:-10, :]


def pt_no_pad(pt):
    return pt[0] - 10, pt[1] - 10


# 我再插入一个函数吧，九宫格找keypoint的？这样可以吗, 要pt在边缘 就减不了5啊。还要pad一下？shi zuixiaozhi ma ?
# 他的黑点是<255,背景白色=没55哦知道了=，但是我看他有些预测的挺好的0.0, 啊，，那就是模型不行哈哈。要该嘛?
# 蜗居的是不是应该直接预测valid然后看错了哪些，错误绿多少
# 不然我没无法评价这个方法好不好，恩，那我来写valid.hha
def pixel_prob_calc(im, keypoints):
    # input: im 3-D, list of cv2.keypoints
    # output: single cv2.keypoints
    if len(keypoints) == 1:
        return keypoints

    biggest = 255
    ps = 5
    single_keypoint = []
    for i in keypoints:
        tmp_im = np.pad(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), [[ps, ps], [ps, ps]], 'symmetric')
        SUM = np.sum(tmp_im[int(i.pt[1] + ps) - ps:int(i.pt[1] + ps) + ps,
                     int(i.pt[0] + ps) - ps: int(i.pt[0] + ps) + ps])  # 这输入法有毒啊，圈加起来也行啊我是想九个点，那就圈加起来吧
        SUM = SUM / (2 * ps) ** 2
        if SUM < biggest:
            biggest = SUM.copy()
            single_keypoint = [i]
    return single_keypoint


# 是这样吗？盼神人呢，应该是这样。


# TODO: ckpt loader not support

def keypoints_gen(pred_image):
    # pred_y is a dict
    # image_name, image, image_size, need_cols, pred_y
    # image_size
    im_pred = {}
    keypoints_ = {}
    for k, col in enumerate(pred_image["need_cols"]):  # col = 'cuff_left_out'
        pred_image_tmp = 255 - cv2.resize(pred_image["pred_y"][0],
                                          pred_image["image_size"][::-1])[:, :, k:k + 1]
        im = pred_image_tmp.copy()

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
        im_pred[col] = im
        detector = cv2.SimpleBlobDetector_create(params)
        # 用什么填，最大值，不对，外圈要和背景颜色一眼才行，先试试看

        keypoints = detector.detect(im_pad(im))
        for i, key in enumerate(keypoints):
            keypoints[i].pt = pt_no_pad(key.pt)
        keypoints_[col] = keypoints
    pred_image["kp_data"] = {"pred": im_pred, "keypoints_": keypoints_}
    return pred_image


def compare_default(pred_image):
    # TODO: 预测结果概率矩阵方差很小则没有预测出来，是一个超参，需要调超参
    # 九宫格找keypoint, keyiba, kanyunle, shishikan
    # 添加keypoints__
    # youyixie还是部队，但是已经好很多了，还有什么操作可以提升。规则吗?太麻烦了吧，是很麻烦
    # 要不然我们先训练万，提交了先，之后慢慢改对三。生成
    # 可以对train进行预测，然后选择train预测自己预测错了的，然后用这种方法减少点，去看他的准确率

    kd = {}
    for col, im in pred_image["kp_data"]["pred"].items():
        keypoints = pred_image["kp_data"]["keypoints_"][col]
        keypoints_update = pixel_prob_calc(im, keypoints)
        kd[col] = keypoints_update
    pred_image["kp_data"]["keypoints__"] = kd
    return pred_image


def draw_keypoint(pred_image, save_path):
    keypoints_ = pred_image["kp_data"]["keypoints_"]
    keypoints__ = pred_image["kp_data"]["keypoints__"]

    fig = plt.figure(figsize=(40, 20))
    for k, (col, kps) in enumerate(pred_image["kp_data"]["keypoints_"].items()):
        fig.add_subplot(4, 8, k * 2 + 1)
        tmp = cv2.drawKeypoints(pred_image["image"], keypoints_[col], np.array([]), (0, 0, 0),
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        plt.title(col + " %s" % len(keypoints_[col]))

        fig.add_subplot(4, 8, k * 2 + 2)
        tmp = cv2.drawKeypoints(pred_image["image"], keypoints__[col], np.array([]), (255, 0, 0),
                                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB))
        plt.title(col + " update %s" % len(keypoints__[col]))

    fig.savefig(save_path)
    plt.close()
    return save_path


parser = argparse.ArgumentParser(description='tf-pose-estimation run')
parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin/ model_path')
parser.add_argument('--testdir', type=str, default='[None]', help='test dir')
parser.add_argument('--outputdir', type=str, default='[None]', help='output dir')
parser.add_argument('--coldir', type=str, default='[None]', help='train dir')
HOME_PATH = "/media/yanpan/7D4CF1590195F939"
# HOME_PATH = "D:"
category = 'dress'
# 'blouse', 'dress', 'outwear', 'skirt', 'trousers'
args = parser.parse_args(
    f"--model {HOME_PATH}/Projects/tf-pose-model/my{category}_prof/tf-pose-3-{category}/graph_freeze.pb "
    "--resolution 368x368 "
    f"--coldir {HOME_PATH}/Projects/fashionai/mytrain/my{category}_prof "  # need col
    f"--outputdir {HOME_PATH}/Projects/fashionai/valid/my{category}_prof/tf-pose-3-{category}/{category} "
    f"--testdir {HOME_PATH}/Projects/fashionai/mytrain/my{category}_prof/val2017".split()
    # f"--testdir {HOME_PATH}/Projects/fashionai/test2/Images/{category}".split()
)

if __name__ == '__main1__':
    image_paths = [args.testdir + "/" + x for x in os.listdir(args.testdir)]
    with open(args.coldir + "/annotations/need_cols.txt", "r", encoding="utf8") as f:
        need_cols = [x.strip() for x in f.readlines()][2:]
    images = []
    for image_path in image_paths:
        image_name = re.sub("^.*(?=Images)", "", image_path)
        image = {}
        image["image_name"] = image_name
        image["image"] = read_img(image_path, None, None)
        image["image_size"] = image["image"].shape[:2]
        image["need_cols"] = need_cols
        images.append(image)

    # images.json export?

    graph = load_graph(args.model)
    pred_images = graph_pred(graph, images, (368, 368))
    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)
    with open(args.outputdir + "/" + "pred_images.npy", "wb") as f:
        np.save(f, pred_images)


def submit(im_args):
    k, pred_image = im_args  # k, pred_image = 0, pred_images[0]
    pred_image_im = keypoints_gen(pred_image)
    # compare_unique: generate pred_image_keypoint_position, single per keypoint
    # pred_image_keypoint_position = compare_unique(pred_image_im)
    pred_image_im_update = compare_default(pred_image_im)

    # detect if pred_image_keypoint_position has more than 1 points
    # 没有画预测的图，啥 ，进步i适合i虎i画的吗， 没有现在预测的图，看不到究竟是啥，，，，？？
    # 就是现在预测只有一个了，及偶没有图了，哦 对啊，那要保存之前的图来对比吧？现在正在覆盖。。
    # 现在根本就没画，没画？不是在跑吗，我只花了>2点的图。这样啊，那要都画吗，
    # chifan hao
    count = {}
    for col, pos in pred_image_im_update["kp_data"]["keypoints_"].items():
        if len(pos) != 1:
            count[col] = len(pos)
    if len(count) > 0:
        logger.info("more than 1 points! %5s %s %s" % (k, pred_image_im_update["image_name"], count))
        save_path = args.outputdir + "/" + pred_image_im_update["image_name"].split("/")[-1]
        draw_keypoint(pred_image_im_update, save_path)

    outcome_dict_tmp = {"id": pred_image_im_update["image_name"],
                        "pos": {col: [i.pt for i in x] for col, x in
                                pred_image_im_update["kp_data"]["keypoints__"].items()}}
    return outcome_dict_tmp


if __name__ == '__main__':
    with open(args.outputdir + "/" + "pred_images.npy", "rb") as f:
        pred_images = np.load(f)

    # pred_images_dict is a list of dicts, every dict contains
    executer = ProcessPoolExecutor(max_workers=7)
    im_args = zip(range(len(pred_images)), pred_images)
    outcome = executer.map(submit, im_args)
    outcome = list(outcome)

    with open(args.outputdir + "/pred.json", "w") as f:
        json.dump(outcome, f)

if __name__ == '__tmp__':
    pic_name = "5cf284e73d397fedee0c10f552e3309c"
    need_pic = [x for x in pred_images if pic_name in x["image_name"]]
    pred_image = need_pic[0]

    # 重叠两个区域，选取覆盖最多的那个像素块：2值化最黑的地方
    # 然后遇到覆盖最多的超过2个，就删除已经被前面找到的那些位置

    # #预测结果，多个keypoints的channel直接采用pt的九宫格平均
    # if __name__ == '__main__':
    #     a = pd.read_csv("/media/yanpan/7D4CF1590195F939/Projects/fashionai/train/Annotations/train.csv", header=0).columns
    #
    #
