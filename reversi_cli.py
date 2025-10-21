import torch
from reversi_game_logic import *
from reversi_gnn_model import ReversiGNN
import json
import os
from datetime import datetime
from reversi_utils import print_board
from config import BOARD_SIZE
from config import MODEL_PATH


def load_model(path=MODEL_PATH):
    model = ReversiGNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def board_to_tensor(board, player):
    return torch.tensor((board * player).flatten(), dtype=torch.float32).unsqueeze(1)


def select_action(model, board, player, edge_index):
    x = board_to_tensor(board, player)
    with torch.no_grad():
        logits = model(x, edge_index)
    mask = torch.zeros_like(logits)
    for r, c in valid_moves(board, player):
        idx = r * BOARD_SIZE + c
        mask[idx] = 1
    logits = logits * mask - (1 - mask) * 1e10
    probs = torch.softmax(logits, dim=0)
    action_idx = torch.multinomial(probs, 1).item()
    return action_idx // BOARD_SIZE, action_idx % BOARD_SIZE


def play_against_ai():
    os.makedirs("logs", exist_ok=True)
    model = load_model()
    edge_index = get_edge_index()
    board = init_board()
    player = 1  # 人間が先手（1）
    history = []

    while not is_game_over(board):
        print_board(board)
        moves = valid_moves(board, player)

        if not moves:
            print(f"Player {player} has no valid moves and must pass.")
            player *= -1
            continue

        if player == 1:
            try:
                r, c = map(int, input("Your move (row col): ").split())
            except Exception:
                print("Invalid input. Please input two integers separated by space.")
                continue
            if (r, c) not in moves:
                print("Invalid move. Try again.")
                continue
        else:
            r, c = select_action(model, board, player, edge_index)
            print(f"AI moves: {r} {c}")

        history.append((board.copy().tolist(), (r, c), player))
        board = make_move(board, (r, c), player)
        player *= -1

    print_board(board)
    winner = (
        "Draw" if board.sum() == 0 else ("You win!" if board.sum() > 0 else "AI wins!")
    )
    print("Game over:", winner)

    log_filename = f"logs/human_vs_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_filename, "w") as f:
        json.dump(history, f)
    print(f"Game log saved to {log_filename}")


if __name__ == "__main__":
    play_against_ai()
