import json
import pandas as pd
import numpy as np


def truth_gen(truth_path):
    with open(truth_path, "r", encoding="utf8") as f:
        data = json.load(f)

    def zip_col(ll):
        tmp_dict = {}
        for k, col in enumerate(need_cols):
            if ll[3 * k + 2] == 2:
                tmp = ll[3 * k:3 * k + 2]
            else:
                tmp = [0, 0]
            tmp_dict[col] = tmp
        return tmp_dict

    keypoints_dict = {x["id"]: zip_col(x["keypoints"]) for x in data["annotations"]}
    image_name_dict = {x["id"]: x["file_name"].split(".")[0] for x in data["images"]}
    kpt_df = pd.DataFrame(keypoints_dict).T
    kpt_index = pd.Series(image_name_dict)
    kpt_df.index = kpt_index.values

    def dist_sk(l, r):
        if l == [0, 0] or r == [0, 0]:
            return 99999999
        return np.sum((np.array(l) - np.array(r)) ** 2) ** 0.5

    kpt_df["sk"] = kpt_df[["armpit_left", "armpit_right"]].T.apply(lambda x: dist_sk(*list(x)))

    return kpt_df


def pred_gen(pred_path):
    with open(pred_path, "r", encoding="utf8") as f:
        data = json.load(f)
    kpt_df = pd.DataFrame(columns=need_cols)
    for imkpt in data:
        im_name = imkpt["id"].split("/")[-1].split(".")[0]
        # xianzai pos <= 1
        im_pos = {col: pos[0] if len(pos) > 0 else [0, 0] for col, pos in imkpt["pos"].items()}

        kpt_df.loc[im_name] = im_pos
    return kpt_df


def dist_visible(l, r):
    # truth == [0,0]
    if l == [0, 0]:
        return 999999999999
    return np.sum((np.array(l) - np.array(r)) ** 2) ** 0.5


def metric_ne(truth, pred):
    """
    (0.105+ 0.141+ 0.176+ 0.212+0.095+ 0.190+ 0.285+ 0.381+ 0.476+ 0.570+0.666)/11

    本次竞赛采用NE (Normalized Error)作为总的比赛排名指标。
    NE: 在只考虑可见点情况下，预测点和标注点平均归一化距离。
    NE = sum(dk/sk*vk)/sum(vk)
    其中 k 为关键点编号，dk 表示预测关键点和标注关键点的距离，
    sk 为距离的归一化参数（上衣、外套、连衣裙为两个腋窝点欧式距离，
    裤子和半身裙为两个裤头点的欧式距离），vk 表示关键点是否可见。
    （遮挡点和不存在的点在竞赛评测中不考虑，所以距离都为0）
    """
    score = {}
    count = {}
    for col in need_cols:
        vk = truth[col] != [0, 0]
        dk = dist_visible(truth[col], pred[col])
        sk = truth["sk"]
        score[col] = dk / sk * vk
        count[col] = vk
    return sum(score.values()) / sum(count.values())


def metric_ne_all(truth_df, pred_df):
    all_score = 0
    all_count = 0
    for im_name in truth_df.index:
        truth, pred = truth_df.loc[im_name].sort_index(), pred_df.loc[im_name].sort_index()
        score = {}
        count = {}
        for col in need_cols:
            vk = truth[col] != [0, 0]
            dk = dist_visible(truth[col], pred[col])
            sk = truth["sk"]
            score[col] = dk / sk * vk
            count[col] = vk
        all_score += sum(score.values())
        all_count += sum(count.values())
    return all_score / all_count


if __name__ == '__main__':
    home_path = "/media/yanpan/7D4CF1590195F939/Projects/fashionai"
    truth_path = f"{home_path}/mytrain/myblouse/annotations/person_keypoints_val2017.json"
    pred_path = f"{home_path}/valid/tf-pose-3-blouse/blouse/pred.json"
    col_path = f"{home_path}/mytrain/myblouse/annotations/need_cols.txt"
    with open(col_path, "r", encoding="utf8") as f:
        need_cols = [x.strip() for x in f.readlines()][2:]

    pred_df = pred_gen(pred_path)
    truth_df = truth_gen(truth_path)
    score_all = metric_ne_all(truth_df, pred_df)
    print(score_all)

    scores = pd.Series({im_name: metric_ne(truth_df.loc[im_name], pred_df.loc[im_name]) for im_name in truth_df.index})
    print(scores.sort_values())
    """
    8182410a941c048f1f50365cae7b6596    0.647701
    d1cd22f264a66ca1bc04c123c4d301b1    0.664920
    d31711aefbf4e86218f47d3e624a2a39    0.670269
    55c44fc4c55fd69c3fa0d4298d1dc42f    0.673277
    425a9a9278389c53690a98ecc260dea1    0.677799
    a32862baff77098cb26e65722a7f8e65    0.680916
    7771c48053633b8932da2fe48db8acce    0.681249
    004fb4dde569db5d748c8f5328e230c6    0.690269
    e8506f8a7b9ee69dbe9125b9446dfe5a    0.696759
    b212ff70d65997356648beb1f23a6132    0.715210
    44c24c324fd7ba4508b31e0ac5a18f4a    0.721080
    dc63be03c0e6f24598671e32720595a0    0.722112
    ba59be6a5b2ded9818a35f3ebeba4725    0.725348
    10e3180b98e3c12786d606d4dd3ab187    0.726537
    3ff96b011eca33ef1e5611361dbba21d    0.819179
    30440d161cd325ed5445aa984bdec0f2    0.844745
    43038b05237534a18831d8406b9d2108    0.854404
    b4a3dc125fccb12aa05547c411e95085    0.855572
    b57cab09491225877b67b07a43b71245    0.922532
    80ad6165f815cc976eff9d44fd41a0ef    0.988153
    bdaaa3f92d8174c702d030a2cfde0e71    1.116625
    8a9dcd8e45a27c629d747bfa43f01907    1.238157
    ad057b6125a0b671383e5164b7384a5b    1.350962
    24eaae82cf80dad3186eb4b6eaaf0bf0    1.489926
    8e20c2046266ee66b100eb273f166768    1.534567
    a3eb8e43cbd1cd486d80aec5d389eac5    1.603260
    bc578726ecfc773ea1fed8573be7b467    1.680018
    57041cd7c3c1abc14bf92566c97b690a    1.699367
    f527913dd1e10b401b043ae524856601    2.769326
    923d61839fb6c2123484af7bb29ec34e    4.246136
    """
