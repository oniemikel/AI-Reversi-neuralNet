import json
import os

TRAIN_INFO_PATH = "./train/train_info.json"


def update_self_train_time(add_num: int) -> None:
    if not os.path.exists(TRAIN_INFO_PATH):
        data = {"self_train_time": 0, "play_train_time": 0}
    else:
        with open(TRAIN_INFO_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

    data["self_train_time"] = data.get("self_train_time", 0) + add_num

    with open(TRAIN_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def update_play_train_time(add_num: int = 1) -> None:
    if not os.path.exists(TRAIN_INFO_PATH):
        data = {"self_train_time": 0, "play_train_time": 0}
    else:
        with open(TRAIN_INFO_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

    data["play_train_time"] = data.get("play_train_time", 0) + add_num

    with open(TRAIN_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
