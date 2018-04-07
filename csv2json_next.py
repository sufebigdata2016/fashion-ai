import json
import matplotlib.image as mpimg  # mpimg 用于读取图片
import numpy as np
import os
import pandas as pd
import shutil


def dataframe2json(annotations_csv, category, image_dir):
    annotations = []
    images = []
    need_cols = annotations_csv.columns[
        np.array([sum(annotations_csv.iloc[:, i] != "-1_-1_-1") for i in range(annotations_csv.shape[1])]) > 0]
    annotations_csv = annotations_csv[need_cols]

    for i in range(len(annotations_csv)):
        tmp = annotations_csv.iloc[i]
        jpg_name = tmp["image_id"].split("/")[-1]
        image_path = image_dir + "/" + tmp["image_id"]
        image = mpimg.imread(image_path)
        if i % 100 == 100 - 1:
            print("reading image %s" % (i + 1))
        images.append({"file_name": jpg_name, "id": i, "height": image.shape[0], "width": image.shape[1]})

        # TODO: image_category
        tmp_keypoints = []
        for data in tmp[2:]:
            data_tmp = []
            for x in data.split("_"):
                # 最后一位有三种情况：-1(不存在该点), 0(该点被遮挡), 1(显示在图中)
                if int(x) == -1:
                    data_tmp.append(0)
                elif int(x) == 0:
                    data_tmp.append(1)
                elif int(x) == 1:
                    data_tmp.append(2)
                else:
                    data_tmp.append(int(x))
            tmp_keypoints += data_tmp
        num_keypoints = len(list(filter(lambda x: x > 0, tmp_keypoints))) // 3
        annotations.append(
            {"num_keypoints": num_keypoints, "keypoints": tmp_keypoints, "image_id": i, "category_id": 1, "id": i})

    categories = [
        {"supercategory": category, "id": 1, "name": category, "keypoints": list(annotations_csv.columns[2:])}]

    json_data_flow = {"images": images, "annotations": annotations, "categories": categories}
    return json_data_flow, need_cols


if __name__ == '__main__':
    file_dir = "/media/yanpan/7D4CF1590195F939/Projects/fashionai"
    train_file_path = "%s/%s" % (file_dir, "train/Annotations/train.csv")
    valid_file_path = "%s/%s" % (file_dir, "train_warmup/Annotations/annotations.csv")
    train_dir = file_dir + "/train"
    valid_dir = file_dir + "/train_warmup"
    category = "dress"
    # 要重新划分训练机和验证机吗? 训练机还要重新划分？现在默认是train的训练和train_warmup的验证 哦，那要整个合在一起，在拆十分之一给验证？
    # blouse也要重新跑，如果换的花，那就重新破吧 不慢，所以用什么模型，用心的模型嘛，那个晓得。旧的模型loss降到多少，新的能将到4
    # 这个不好说，主要是还没有比较新旧之间的差别
    # 800多个>2的点不好弄。400多个也不好浓。 我们就按照每个点旁边的九宫格求和来吧，哪个大区那个
    # 九宫格是啥? keyPiont 不是是一个点吗，然后他旁边不是还有八个点，求个和
    # 旁边的那里八个? 就是im，是个灰度图，上面的每一个点不是被八个点包围吗。就九宫格-。-？?？
    # a 中间那个点 是被8个点包围，我们用九个点来看那个keypoint概率更大
    # keypoint不是一个圆圈吗? 那个我不知道怎么把这个那个圆圈取出来啊。啊，好难啊，我想先看看pred的800
    # 怎么这么多，我在看图
    # 1. 没有pad，有些还是0
    # 2. 对于那些点多的，直接取圈最大的(这个?这个不行啊，圈是用blobdetector找出来的，我觉得是看里面的直，不是圈的大小，里面的之是概率嘛，就是这里我刚才说用九宫格求一个综合
    # 根据峰度还是均值，均值。均值不行吧。峰值小的峰，均值就比较小？我对detect算法一窍不通。我也不同。。要不随即取一个吧。。。。那怎么可以。我觉得九宫格可以啊。好。那我们先开始ba
    # 别的衣服训练了吧，数据v机有点乱。）
    # 3. 有些点可以用相对位置判断，有很多case,很烦
    # 先写一个pad,pad是什么//就是有的是0,然后在边缘，他没有detect出来哦哦。好我去改胰腺癌
    # 'blouse', 'dress', 'outwear', 'skirt', 'trousers'
    import numpy as np
    a =np.ones((3,3))
    annotations_csv_train = pd.read_csv(train_file_path, encoding="utf8")
    annotations_csv_train = annotations_csv_train[annotations_csv_train["image_category"] == category]
    annotations_csv_valid = pd.read_csv(valid_file_path, encoding="utf8")
    annotations_csv_valid = annotations_csv_valid[annotations_csv_valid["image_category"] == category]

    annotations_csv_train = annotations_csv_train.sample(frac=1, random_state=0)

    fdir = f"{file_dir}/mytrain/my{category}"
    tp_train = f"{fdir}/train2017"
    tp_val = f"{fdir}/val2017"
    annot = f"{fdir}/annotations"

    for tdir in [fdir, tp_train, tp_val, annot]:
        os.mkdir(tdir) if not os.path.exists(tdir) else 0

    for train_file in annotations_csv_train["image_id"]:
        fp = "%s/%s" % (train_dir, train_file)
        shutil.copy(fp, tp_train)
    for valid_file in annotations_csv_valid["image_id"]:
        fp = "%s/%s" % (valid_dir, valid_file)
        shutil.copy(fp, tp_val)
    # annotations_csv_valid = annotations_csv
    # annotations_csv_valid = annotations_csv

    json_data_flow_train, need_cols = dataframe2json(annotations_csv_train, category, train_dir)
    json_data_flow_valid, _ = dataframe2json(annotations_csv_valid, category, valid_dir)

    with open(f"{annot}/person_keypoints_train2017.json", "w", encoding="utf8") as f:
        json.dump(json_data_flow_train, f, ensure_ascii=False)
    with open(f"{annot}/person_keypoints_val2017.json", "w", encoding="utf8") as f:
        json.dump(json_data_flow_valid, f, ensure_ascii=False)
    with open(f"{annot}/need_cols.txt", "w", encoding="utf8") as f:
        f.writelines([x + "\n" for x in need_cols])

    TASK_KEY_POINTS = len(need_cols) - 2 + 1  # backgroud is one category
    with open("./src/taskdesc.py", "w", encoding="utf8") as f:
        f.writelines("TASK_KEY_POINTS = %s" % TASK_KEY_POINTS)
