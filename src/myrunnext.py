# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:04:00 2018

@author: zdr2535
"""
aaaaaa = ["neckline_left", "neckline_right", "center_front", "shoulder_left", "shoulder_right", "armpit_left",
          "armpit_right", "cuff_left_in", "cuff_left_out", "cuff_right_in", "cuff_right_out", "top_hem_left,",
          "top_hem_right"]

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


# We use our "load_graph" function
graph = load_graph("/media/yanpan/7D4CF1590195F939/Projects/tf-pose-model/myblouse/tf-pose-1-blouse/graph_freeze.pb")

# We can verify that we can access the list of operations in the graph
for op in graph.get_operations():
    print(op.name)  # <--- printing the operations snapshot below
    # prefix/Placeholder/inputs_placeholder
    # ...
    # prefix/Accuracy/predictions

# We access the input and output nodes
x = graph.get_tensor_by_name('prefix/image:0')
y = graph.get_tensor_by_name('prefix/Openpose/concat_stage7:0')

image_size = []
# We launch a Session
with tf.Session(graph=graph) as sess:
    test_features = cv2.imread(
        '/media/yanpan/7D4CF1590195F939/Projects/fashionai/myblouse/val2017/1ae7c342e84270a8c1c7803a403dddaf.jpg')
    image_size = test_features.shape[:2]
    test_features_tmp = cv2.resize(test_features, (368, 368))[np.newaxis, :, :, :].astype(np.float32)
    # compute the predicted output for test_x
    pred_y = sess.run(y, feed_dict={x: test_features_tmp})
    # print(pred_y)

fig = plt.figure(figsize=(20, 20))
index = 13
# pred_y = np.load('./pred_y.npy')
dic = {}
for i in range(index):
    pred_y_tmp = 255 - cv2.resize(pred_y[0], image_size[::-1])[:, :, i:i + 1]

    # plt.figure(figsize=(12, 12))
    # plt.imshow(pred_y[:, :, 0])

    # Read image
    # im = cv2.imread('D:/github/tf-pose-estimation/images/0faaa01d3ae07334448b8fe7643febc8.jpg')
    im = pred_y_tmp.copy()


    def float2unit8(im):
        return (255 * (im - np.min(im)) / (np.max(im) - np.min(im))).astype(np.uint8)


    im = float2unit8(im)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 15

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im)
    axis = [n.pt for n in keypoints]
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    dic[i] = axis

    a = fig.add_subplot(4, 4, i + 1)
    ttt = 255 - im_with_keypoints + cv2.cvtColor(test_features, cv2.COLOR_BGR2RGB)
    plt.imshow(ttt / ttt.max(axis=(0,1)) * 255)
    plt.title(aaaaaa[i])

plt.savefig("mypng.png")
# Show blobs
# cv2.imshow("Keypoints", im_with_keypoints)
# cv2.waitKey(0)

# for i in range(index):
#
#     plt.show()
#
# kkk = pred_y[0, :, :, :14].reshape(46 * 46, 14)
#
# plt.imshow(20-np.argmax(kkk, axis=1).reshape(46, 46))
# plt.imshow(test_features[0])
#
#
# plt.imshow(test_features[0][:,:,0]//150+cv2.resize(np.max(pred_y[0, :, :, :13],axis=2),test_features[0].shape[:2]))
# plt.imshow(np.max(pred_y[0, :, :, 13:14],axis=2))
