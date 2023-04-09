import json

import pandas
from sklearn.datasets import fetch_20newsgroups
import pandas as pd
import openai
import os

PROXY_URL = "xxxxx"
os.environ["http_proxy"] = PROXY_URL
os.environ["https_proxy"] = PROXY_URL
OPENAI_API_KEY = "xxxxxxxx"


def create_examples(train_rate=0.9):
    """
    :param train_rate: 默认时90%的样本用于训练,10%用于评估
    :return:
    """
    # 二分类问题
    class_names = ['rec.sport.baseball', 'rec.sport.hockey']
    # 自动下载开源分类数据集到 data文件夹，如果数据已经存在则从data目录直接读取
    sports_dataset = fetch_20newsgroups(subset='train', shuffle=True, random_state=42, categories=class_names, data_home="data")
    # print(sports_dataset['data'][0]) #print record
    # 分别统计样本总量，每个类别对应数量
    len_all, len_baseball, len_hockey = len(sports_dataset.data), len([e for e in sports_dataset.target if e == 0]), len(
        [e for e in sports_dataset.target if e == 1])
    print(f"Total examples: {len_all}, Baseball examples: {len_baseball}, Hockey examples: {len_hockey}")
    # 对训练目标的类别名称简化处理，如 rec.sport.baseball ==> baseball
    labels = [sports_dataset.target_names[x].split('.')[-1] for x in sports_dataset['target']]
    # 读取输入训练的文本
    texts = [text.strip() for text in sports_dataset['data']]
    # 将Input(文本-prompt)和target(classification)字段合并为openai要求的格式(prompt和completion)
    df = pandas.DataFrame(zip(texts, labels), columns=['prompt', 'completion'])
    # print(df.head())
    # 样本shuffle
    df = df.sample(frac=1)
    # 根据训练样本占有率计算训练样本数量
    train_count = int(len_all * train_rate)
    # 训练样本
    train_df = df[0:train_count]
    # 验证集样本
    valid_df = df[train_count:]
    # 分别存储为文件
    train_df.to_json("data/train_sport.jsonl", orient='records', lines=True, force_ascii=False)

    valid_df.to_json("data/valid_sport.jsonl", orient='records', lines=True, force_ascii=False)
    print("create train: {}, valid: {} examples success...".format(train_count, len(valid_df)))


def create_train_task():
    import warnings
    warnings.simplefilter("ignore", UserWarning)
    openai.api_key = OPENAI_API_KEY
    openai.verify_ssl_certs = False
    openai.proxy = PROXY_URL
    train_file_result = openai.File.create(
        file=open("data/train_sport.jsonl"),
        purpose="fine-tune"
    )
    """
    result:
        {
        "bytes": 1968569,
        "created_at": 1681022647,
        "filename": "file",
        "id": "file-P9cPrsclwq6l0aRc02RV1QaT",
        "object": "file",
        "purpose": "fine-tune",
        "status": "uploaded",
        "status_details": null
        }
    """
    print(train_file_result["id"])
    valid_file_result = openai.File.create(
        file=open("data/valid_sport.jsonl"),
        purpose="fine-tune"
    )
    print(valid_file_result["id"])
    result = openai.FineTune.create(
        model="davinci",  # 目前允许fine-tune的模型只有(ada, babbage, curie, davinci)
        training_file=train_file_result["id"],  # 将train上传文件的id作为训练目标
        validation_file=valid_file_result["id"],  # 将valid上传文件的id作为评估目标
        n_epochs=4,
        batch_size=128,
        learning_rate_multiplier=0.2,
        prompt_loss_weight=0.01,
        compute_classification_metrics=True,
        classification_positive_class="baseball",  # 由于是二分类问题，因此此处将'baseball'作为正分类（1），默认hockey作为negative分类(0)，
        suffix="sport"  # 对训练的模型添加后缀便于区分
    )
    with open("data/result.dat", "w+") as writer:
        json.dump(result, writer)


def query_train_result():
    import warnings
    warnings.simplefilter("ignore", UserWarning)
    openai.api_key = OPENAI_API_KEY
    openai.verify_ssl_certs = False
    openai.proxy = PROXY_URL
    with open("data/result.dat", "r+") as reader:
        data = json.load(reader)
        request_id = data["id"]
        # result=openai.FineTune.list(api_key=OPENAI_API_KEY,request_id=request_id)
        # print(result)
        # 查询模型调优状态和结果
        result = openai.FineTune.retrieve(request_id)
        print(result)


if __name__ == "__main__":
    # create_train()
    # post_train()
    query_train_result()
