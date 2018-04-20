import argparse

parser = argparse.ArgumentParser(description='task desc keypoints write')
parser.add_argument('--keypoints', default='1', help='model name')

if __name__ == '__main__':
    args = parser.parse_args()

    with open("taskdesc.py", "w", encoding="utf8") as f:
        f.writelines("TASK_KEY_POINTS = %s" % args.keypoints)

    # 14, 16, 15, 5, 8

if __name__ == "__test__":

    # t, q = sess.run([tf.concat(l1, axis=0), q_vect_split[0]])
    # t, q = sess.run([tf.concat(l2, axis=0), q_heat_split[0]])

    t, q = sess.run([tf.concat(l2, axis=0), q_heat])

    import matplotlib.pyplot as plt

    for i in range(32):
        fig = plt.figure(figsize=(40, 20))
        for k in range(14):
            fig.add_subplot(4, 8, k * 2 + 1)
            plt.imshow(t[i, :, :, k])
            plt.title("t %s:%s" % (i, k))

            fig.add_subplot(4, 8, k * 2 + 2)
            plt.imshow(q[i, :, :, k])
            plt.title("q %s:%s" % (i, k))

        fig.savefig("pp%s.png" % i)
        plt.close()

    import tensorflow as tf
    import matplotlib.pyplot as plt
    import numpy as np

    sess = tf.Session(config=tf.ConfigProto())
    a = np.random.randint(0, 255, (32, 96, 46, 14))
    x = np.random.randint(0, 255, (32, 96, 46, 14))
    A = tf.constant(a)
    X = tf.constant(x)

    b = tf.reshape(A, (32, 96 * 46, 14))
    c = tf.argmax(b, 1)
    c1 = c // 46
    c2 = c % 46

    y = tf.reshape(X, (32, 96 * 46, 14))
    z = tf.argmax(y, 1)
    z1 = z // 46
    z2 = z % 46

    s = tf.sqrt(tf.to_float((z1 - c1) ** 2 + (z2 - c2) ** 2))
    s0 = tf.reduce_sum(s)
    sess.run(s0)
    plt.imshow(a[0, :, :, 0] == np.max(a[0, :, :, 0]))
    plt.imshow(x[0, :, :, 0] == np.max(x[0, :, :, 0]))
    plt.show()


    def max_l2_dist(a, b, name=None):
        a0 = tf.argmax(tf.reshape(a, (a.shape[0], a.shape[1] * a.shape[2], a.shape[3])), 1)
        b0 = tf.argmax(tf.reshape(b, (b.shape[0], b.shape[1] * b.shape[2], b.shape[3])), 1)
        s = tf.sqrt(tf.to_float((a0 // a.shape[2] - b0 // b.shape[2]) ** 2 + (a0 % a.shape[2] - b0 % b.shape[2]) ** 2))
        return tf.reduce_sum(s, name=name)


    sess.run(max_l2_dist(A, X))
