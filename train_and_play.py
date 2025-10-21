from reversi_trainer import train_gnn
from reversi_cli import play_against_ai
from reversi_gui import main as play_with_gui, ReversiVisualizer
from config import NUM_SELF_PLAY_GAMES


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
