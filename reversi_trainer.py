import os
import torch
import torch.nn.functional as F
import numpy as np

from reversi_game_logic import (
    init_board,
    valid_moves,
    make_move,
    is_game_over,
    get_edge_index,
)
from reversi_gnn_model import ReversiGNN
from config import MODEL_PATH
from reversi_utils import print_board
from elo_rating import EloRating, update_elo_from_match_results


def board_to_tensor(board, player):
    return torch.tensor((board * player).flatten(), dtype=torch.float32).unsqueeze(1)


def select_action(model, board, player, edge_index):
    x = board_to_tensor(board, player)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)

    assert (
        logits.dim() == 1 and logits.shape[0] == 64
    ), "logitsの形状が異常です。64次元であるべき。"

    moves = valid_moves(board, player)
    if not moves:
        return None

    mask = torch.zeros_like(logits)
    for r, c in moves:
        idx = r * 8 + c
        mask[idx] = 1

    masked_logits = logits.clone()
    masked_logits[mask == 0] = -1e9

    max_val = masked_logits.max()
    if max_val < -1e8:
        masked_logits = torch.where(
            mask.bool(),
            torch.zeros_like(masked_logits),
            torch.full_like(masked_logits, -1e9),
        )
    else:
        masked_logits -= max_val

    probs = torch.softmax(masked_logits, dim=0)

    if torch.isnan(probs).any() or probs.sum() < 1e-8:
        uniform_probs = torch.zeros_like(probs)
        uniform_probs[mask == 1] = 1.0
        uniform_probs /= uniform_probs.sum()
        probs = uniform_probs

    action_idx = torch.multinomial(probs, 1).item()
    return action_idx // 8, action_idx % 8


def train_gnn(num_games=1000, print_every=1, external_history=None, visualizer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReversiGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    edge_index = get_edge_index().to(device)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # --- 人間対戦など外部履歴からの追加学習 ---
    if external_history is not None:
        model.train()
        for x, action_idx, player in external_history:
            x = x.to(device)
            optimizer.zero_grad()
            logits = model(x, edge_index)
            loss = F.cross_entropy(
                logits.unsqueeze(0), torch.tensor([action_idx], device=device)
            )
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model updated from external history and saved to {MODEL_PATH}")
        return model

    # --- 自己対戦学習ループ ---
    for game in range(num_games):
        board = init_board()
        player = 1

        while not is_game_over(board):
            moves = valid_moves(board, player)
            if not moves:
                player *= -1
                continue

            x = board_to_tensor(board, player).to(device)
            model.train()
            logits = model(x, edge_index)

            # 有効手の中から最も確率の高い行動を選択
            logits_np = logits.cpu().detach().numpy()
            move_indices = [r * 8 + c for r, c in moves]
            mask = np.full(64, -1e9, dtype=np.float32)
            for idx in move_indices:
                mask[idx] = logits_np[idx]
            probs = F.softmax(torch.tensor(mask), dim=0).numpy()
            action_idx = np.random.choice(64, p=probs)
            r, c = divmod(action_idx, 8)

            # 損失と更新
            optimizer.zero_grad()
            loss = F.cross_entropy(
                logits.unsqueeze(0), torch.tensor([action_idx], device=device)
            )
            loss.backward()
            optimizer.step()

            board = make_move(board, (r, c), player)
            player *= -1

        # GUIがあれば最終盤面を更新（毎手でなく1局単位で）
        if visualizer:
            visualizer.update(board)

        if (game + 1) % print_every == 0:
            print(f"Game {game+1} / {num_games}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def evaluate_models(model_a, model_b, num_games=20):
    """
    model_a と model_b を対戦させて結果を返す。
    結果は model_a の視点からの勝ち(1), 引き分け(0.5), 負け(0) のリスト。
    """

    results = []

    for _ in range(num_games):
        board = init_board()
        player = 1  # 先手は model_a とする

        while not is_game_over(board):
            moves = valid_moves(board, player)
            if not moves:
                player *= -1
                continue

            if player == 1:
                action = select_action(model_a, board, player, get_edge_index())
            else:
                action = select_action(model_b, board, player, get_edge_index())

            if action is None:
                player *= -1
                continue

            board = make_move(board, action, player)
            player *= -1

        # 勝敗判定
        black_count = np.sum(board == 1)
        white_count = np.sum(board == -1)
        if black_count > white_count:
            results.append(1 if player == -1 else 0)  # playerが反転しているため逆転注意
        elif white_count > black_count:
            results.append(0 if player == -1 else 1)
        else:
            results.append(0.5)

    return results


def train_loop(
    total_cycles=50,
    games_per_cycle=20,
    eval_games=20,
    print_every=1,
):
    elo = EloRating()
    elo.add_player("baseline", 1200)
    elo.add_player("current", 1200)

    # 最初のbaselineモデルを保存（空モデル or 既存モデル）
    if os.path.exists(MODEL_PATH):
        baseline_model = ReversiGNN()
        baseline_model.load_state_dict(torch.load(MODEL_PATH))
    else:
        baseline_model = ReversiGNN()
        torch.save(baseline_model.state_dict(), MODEL_PATH)

    current_model = ReversiGNN()
    current_model.load_state_dict(torch.load(MODEL_PATH))

    for cycle in range(1, total_cycles + 1):
        print(f"===== Training cycle {cycle} =====")

        # 学習
        current_model = train_gnn(num_games=games_per_cycle, print_every=print_every)

        # baselineモデルは直前のcurrent_modelを保存したものを使う
        torch.save(current_model.state_dict(), MODEL_PATH)

        # 評価
        baseline_model.load_state_dict(torch.load(MODEL_PATH))

        results = evaluate_models(current_model, baseline_model, num_games=eval_games)

        update_elo_from_match_results(elo, "current", "baseline", results)

        print(f"[Cycle {cycle}] ELO Ratings: {elo.ratings}")

        # 進捗保存
        torch.save(current_model.state_dict(), MODEL_PATH)
        elo.save("logs/elo_ratings.json")

    print("Training loop finished.")
