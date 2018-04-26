import argparse
import logging

import tensorflow as tf

from networks import get_network

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True

if __name__ == '__main__':
    """
    Use this script to just save graph and checkpoint.
    While training, checkpoints are saved. You can test them with this python code.
    """
    parser = argparse.ArgumentParser(description='Tensorflow Pose Estimation Graph Extractor')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet / mobilenet_thin/ model_path')
    parser.add_argument('--pb_path', default='./tmp/graph.pb', help='file path to save pb file')
    parser.add_argument('--resolution', default='368x368', help='368x368')
    parser.add_argument('--save_path', default=None, help="path to save ckpt")
    parser.add_argument('--save_step', default=1, help="step to save ckpt")
    args = parser.parse_args()

    x, y = args.resolution.split("x")
    file_dir = "/".join(args.pb_path.split("/")[:-1])
    file_name = args.pb_path.split("/")[-1]
    input_node = tf.placeholder(tf.float32, shape=(None, int(x), int(y), 3), name='image')

    with tf.Session(config=config) as sess:
        net, _, last_layer = get_network(args.model, input_node, sess, trainable=False)

        tf.train.write_graph(sess.graph_def, file_dir, file_name, as_text=True)

        if args.save_path:
            graph = tf.get_default_graph()
            dir(graph)
            for n in tf.get_default_graph().as_graph_def().node:
                if 'concat_stage' not in n.name:
                    continue
                print(n.name)

            saver = tf.train.Saver(max_to_keep=100)
            saver.save(sess, args.save_path, global_step=args.save_step)
