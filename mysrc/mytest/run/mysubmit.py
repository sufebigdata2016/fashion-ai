import json
import pandas as pd
import re


def pred_gen(pred_path, category, need_cols):
    with open(pred_path, "r", encoding="utf8") as f:
        data = json.load(f)
    kpt_df = pd.DataFrame(columns=need_cols)
    for k, imkpt in enumerate(data):
        # xianzai pos <= 1
        im_pos = {col: "%s_%s_1" % (round(pos[0][0]), round(pos[0][1])) if len(pos) > 0 else "0_0_0" for col, pos in
                  imkpt["pos"].items()}
        im_pos["image_category"] = category
        im_pos["image_id"] = re.sub("^.*(?=Image)", "", imkpt["id"])
        kpt_df.loc[k] = im_pos
    return kpt_df


HOME_PATH = "/media/yanpan/7D4CF1590195F939/Projects/fashionai"
# HOME_PATH = "D:/Projects/fashionai"
categories = ['blouse', 'dress', 'outwear', 'skirt', 'trousers']

categories_df = []
for category in categories:
    output_dir = f"{HOME_PATH}/pred2/my{category}_prof/tf-pose-1-{category}/{category}"
    need_col_dir = f"{HOME_PATH}/mytrain/my{category}_prof/annotations/need_cols.txt"
    json_file = output_dir + "/pred.json"
    with open(need_col_dir, "r", encoding="utf8") as f:
        need_cols = [x.strip() for x in f.readlines()]
    try:
        category_df = pred_gen(json_file, category, need_cols)
        categories_df.append(category_df)
    except Exception as e:
        print("pred_gen error", e.__class__, e.__context__)

cdf = pd.concat(categories_df, axis=0)
cdf = cdf.fillna("-1_-1_-1").reset_index(drop=True)

test = pd.read_csv(f"{HOME_PATH}/test2/test.csv", encoding="utf8")
train = pd.read_csv(f"{HOME_PATH}/train/Annotations/train.csv", encoding="utf8")
all_cols = train.columns

if len(cdf) != len(test):
    print("some pic missing")

pred_file = f"{HOME_PATH}/pred2/my_prof.csv"
cdf = pd.merge(test, cdf, on=list(test.columns), how="inner")[all_cols]
cdf = cdf.fillna("-1_-1_-1")
cdf[all_cols].to_csv(pred_file, index=False)
