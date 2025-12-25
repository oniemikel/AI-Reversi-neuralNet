from reversi_trainer import train_gnn_one_game
from reversi_cli import play_against_ai
from reversi_gnn_model import ReversiGNN
from reversi_gui import main as play_with_gui, ReversiVisualizer
from config import NUM_SELF_PLAY_GAMES, MODEL_PATH
import json
import os
import time
import torch

TRAIN_INFO_PATH = "./train/train_info.json"

def update_self_train_time(add_num: int) -> None:
    if not os.path.exists(TRAIN_INFO_PATH):
        data = {"self_train_time": 0}
    else:
        with open(TRAIN_INFO_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

    data["self_train_time"] = data.get("self_train_time", 0) + add_num

    with open(TRAIN_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def main():
    while True:
        try:
            n = input("How many times do you want to train? (default: 1000): ")
            n = int(n) if n.strip() else NUM_SELF_PLAY_GAMES
            break
        except ValueError:
            print("Please input a valid INTEGER.")

    while True:
        mode = input("Do you want to train with GUI visualization? (y/n): ").lower()
        if mode in ("y", "n"):
            use_gui = mode == "y"
            break
        print("Please enter 'y' or 'n'.")

    visualizer = ReversiVisualizer() if use_gui else None

    print(f"Starting training ({n} games)...")

    # 前回のモデルがあれば読み込み
    if os.path.exists(MODEL_PATH):
        model = ReversiGNN()
        model.load_state_dict(torch.load(MODEL_PATH))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        print("Loaded previous model.")
    else:
        model = None
        optimizer = None

    for i in range(1, n + 1):
        print(f"[{i}/{n}] training...")
        model, optimizer = train_gnn_one_game(
            model=model, optimizer=optimizer, visualizer=visualizer
        )

        # ★ 毎ゲーム後にモデルを保存
        torch.save(model.state_dict(), MODEL_PATH)

        # 学習回数も保存
        update_self_train_time(1)

        time.sleep(0.05)

    print("Training finished.")

    while True:
        v = input("Do you want to play with GUI? (y/n): ").lower()
        if v == "y":
            play_with_gui()
            break
        elif v == "n":
            play_against_ai()
            break
        print("Please enter 'y' or 'n'.")


if __name__ == "__main__":
    main()
