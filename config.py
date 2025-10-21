import json
import os

# config.jsonのパス
CONFIG_JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config.json"
)

# JSONファイル読み込み
with open(CONFIG_JSON_PATH, "r") as f:
    cfg = json.load(f)

# ROOT_DIRを絶対パスに変換（このconfig.pyのある場所を基準に）
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg["ROOT_DIR"])
)

# 各パスを絶対パスに変換
MODEL_PATH = os.path.join(ROOT_DIR, *cfg["MODEL_PATH"].split('/'))
LOG_DIR = os.path.join(ROOT_DIR, cfg["LOG_DIR"])
TRAIN_DIR = os.path.join(ROOT_DIR, cfg["TRAIN_DIR"])

# ディレクトリがなければ作成
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)

# 学習パラメータ
NUM_SELF_PLAY_GAMES = cfg["NUM_SELF_PLAY_GAMES"]
NUM_EPOCHS = cfg["NUM_EPOCHS"]
LEARNING_RATE = cfg["LEARNING_RATE"]
BOARD_SIZE = cfg["BOARD_SIZE"]
MODEL_HIDDEN_CHANNELS = cfg["MODEL_HIDDEN_CHANNELS"]
MODEL_OUTPUT_DIM = cfg["MODEL_OUTPUT_DIM"]
