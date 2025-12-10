from reversi_trainer import train_gnn
from reversi_cli import play_against_ai
from reversi_gui import main as play_with_gui, ReversiVisualizer
from config import NUM_SELF_PLAY_GAMES
import json
import os

TRAIN_INFO_PATH = "./train/train_info.json"


def update_self_train_info(add_num: int) -> None:
    """train_info.json の self_train_info を add_num 分だけ加算する"""
    # ファイルがない場合は初期化
    if not os.path.exists(TRAIN_INFO_PATH):
        data = {"self_train_info": 0}
    else:
        with open(TRAIN_INFO_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

    # 値を加算
    data["self_train_info"] = data.get("self_train_info", 0) + add_num

    # 保存
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
        gui_mode = input("Do you want to train with GUI visualization? (y/n): ").lower()
        if gui_mode in ("y", "n"):
            use_gui = gui_mode == "y"
            break
        print("Please enter 'y' or 'n'.")

    print(f"Starting training ({n} games)...")

    visualizer = ReversiVisualizer() if use_gui else None
    train_gnn(num_games=n, visualizer=visualizer)
    
    visualizer.window.close()

    update_self_train_info(n)
    
    print("Training completed. Now starting human vs AI game.")
    while True:
        c = input("Do you want to play with GUI? (y/n): ").lower()
        if c == "y":
            play_with_gui()
            break
        elif c == "n":
            play_against_ai()
            break
        print("Please enter 'y' or 'n'.")


if __name__ == "__main__":
    main()
