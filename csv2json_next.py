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
    category = "blouse"
    # 'blouse', 'dress', 'outwear', 'skirt', 'trousers'
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
