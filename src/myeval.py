import json
import pandas as pd
import numpy as np

category = "blouse"


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

    if category in ["blouse", "dress", "outwear"]:
        kpt_df["sk"] = kpt_df[["armpit_left", "armpit_right"]].T.apply(lambda x: dist_sk(*list(x)))
    elif category in ['skirt', 'trousers']:
        kpt_df["sk"] = kpt_df[["waistband_left", "waistband_right"]].T.apply(lambda x: dist_sk(*list(x)))

    return kpt_df


def pred_gen(pred_path):
    with open(pred_path, "r", encoding="utf8") as f:
        data = json.load(f)
    kpt_df = pd.DataFrame(columns=need_cols)
    for imkpt in data:
        im_name = imkpt["id"].split("/")[-1].split(".")[0]
        # xianzai pos <= 1
        im_pos = {col: [pos[0][0], pos[0][1]] if len(pos) > 0 else [0, 0]
                  for col, pos in imkpt["pos"].items()}

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


def metric_ne_all(truth_df, pred_df, detail=False):
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
    if detail:
        return {"ne": all_score / all_count, "score": all_score, "count": all_count}
    else:
        return {"ne": all_score / all_count}


if __name__ == '__main__':
    category = "dress"
    # 'blouse', 'dress', 'outwear', 'skirt', 'trousers'
    home_path = "/media/yanpan/7D4CF1590195F939/Projects/fashionai"
    truth_path = f"{home_path}/mytrain/my{category}_prof/annotations/person_keypoints_val2017.json"
    pred_path = f"{home_path}/valid/my{category}_prof/tf-pose-2-{category}/{category}/pred.json"
    col_path = f"{home_path}/mytrain/my{category}_prof/annotations/need_cols.txt"
    with open(col_path, "r", encoding="utf8") as f:
        need_cols = [x.strip() for x in f.readlines()][2:]

    pred_df = pred_gen(pred_path)
    truth_df = truth_gen(truth_path)
    result_dict = metric_ne_all(truth_df, pred_df, True)
    # print(result_dict["score"], result_dict["count"])
    print(result_dict)

    """
    wrong !
    blouse
    5:  0.03284665363277248       0.03284545224978625
    10: 
    15: 0.03278980634911917       0.033033534870904366
    20: 
    30: 
    
    dress
    5:  0.08860700258268969      0.04694559976792134
    10: 0.08792957300689651
    15: 0.08698836921054152      0.047236180801040466
    20: 0.08694636300859725
    25: 0.08675139580577107
    30: 0.08612128586206062
    40: 0.08608220639997487
    50: 0.0854804998513105
    
    outwear
    5:  0.036788281075332656
    10: 0.035710530346023235
    15: 0.03563534374157036
    20: 0.03564851505924366
    30: 0.035652795612567574
    
    skirt
    5:  0.1270994314524178
    10: 0.12618443761807832
    15: 0.1259053856757867
    20: 0.12529998787517646
    30: 0.12495066969959043
    
    trousers
    5:  0.10676306262255453
    10: 0.10662648496058096
    15: 0.10624003376478061
    20: 0.10633474500899222
    30: 0.10644685099599915
    
    九宫格最xiao值 0.07547121297895418 (0.0620810531643999 * 10888 + 0.08698836921054152 * 8223 + 0.03563534374157036 * 7523 + 0.1259053856757867 * 3466 + 0.10624003376478061 * 5719) / (10888 + 8223 + 7523 + 3466 + 5719)
    """

    scores = pd.Series({im_name: metric_ne(truth_df.loc[im_name], pred_df.loc[im_name]) for im_name in truth_df.index})
    print(scores.sort_values())
