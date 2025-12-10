# reversi_trainer.py
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
from config import MODEL_PATH, BOARD_SIZE
from reversi_utils import print_board
from elo_rating import EloRating, update_elo_from_match_results


def board_to_tensor(board, player):
    """
    board: numpy array (BOARD_SIZE, BOARD_SIZE)
    player: 1 or -1
    returns: torch.Tensor([num_nodes, in_channels]) float32
    """
    return torch.tensor((board * player).flatten(), dtype=torch.float32).unsqueeze(1)


def select_action(model, board, player, edge_index, device="cpu", deterministic=False):
    """
    model: ReversiGNN
    board: numpy array
    player: 1 or -1
    deterministic: True -> argmax (評価時)、False -> sampling (学習時)
    returns: (r, c) or None if no moves
    """
    x = board_to_tensor(board, player).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index.to(device))

    assert logits.dim() == 1 and logits.shape[0] == BOARD_SIZE * BOARD_SIZE, "logits の形状が異常です。"

    moves = valid_moves(board, player)
    if not moves:
        return None

    # mask illegal moves
    mask = torch.zeros_like(logits, dtype=torch.bool)
    for r, c in moves:
        idx = r * BOARD_SIZE + c
        mask[idx] = True

    masked_logits = logits.clone()
    masked_logits[~mask] = -1e9

    if deterministic:
        action_idx = int(torch.argmax(masked_logits).item())
    else:
        # numerical stability
        m = masked_logits - masked_logits.max()
        probs = torch.softmax(m, dim=0)
        if torch.isnan(probs).any() or probs.sum() < 1e-8:
            # fallback uniform over legal
            probs = torch.zeros_like(probs)
            probs[mask] = 1.0
            probs = probs / probs.sum()
        action_idx = int(torch.multinomial(probs, 1).item())

    return action_idx // BOARD_SIZE, action_idx % BOARD_SIZE


def train_gnn(num_games=1000, print_every=10, external_history=None, visualizer=None):
    """
    REINFORCE ベースの自己対戦学習を行う実装。
    - external_history: [(x_tensor, action_idx, player), ...] を渡すと模倣学習的に1局分だけ更新する（GUI微調整用）
    - 戦略: 各局面での log_prob を蓄え、ゲーム終了時に勝敗に基づく報酬で一括更新する（モンテカルロ方策勾配）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ReversiGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    edge_index = get_edge_index().to(device)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    # external_history による微調整（既存挙動を保つ）
    if external_history is not None:
        model.train()
        for x, action_idx, player in external_history:
            x = x.to(device)
            optimizer.zero_grad()
            logits = model(x, edge_index)
            loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([action_idx], device=device))
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Model updated from external history and saved to {MODEL_PATH}")
        return model

    # --- 自己対戦学習（REINFORCE） ---
    for game in range(num_games):
        board = init_board()
        player = 1

        # 一局分の記録
        log_probs = []
        actions = []
        players = []

        while not is_game_over(board):
            moves = valid_moves(board, player)
            if not moves:
                player *= -1
                continue

            x = board_to_tensor(board, player).to(device)
            model.train()
            logits = model(x, edge_index)

            # マスクして確率を得る
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for r, c in moves:
                idx = r * BOARD_SIZE + c
                mask[idx] = True

            masked_logits = logits.clone()
            masked_logits[~mask] = -1e9
            m = masked_logits - masked_logits.max()
            probs = torch.softmax(m, dim=0)

            # 防御: nan または sum が小さい場合は均等に
            if torch.isnan(probs).any() or probs.sum() < 1e-8:
                probs = torch.zeros_like(probs)
                probs[mask] = 1.0
                probs = probs / probs.sum()

            # サンプリングして行動選択（学習時は確率的に）
            action_idx = int(torch.multinomial(probs, 1).item())
            log_prob = torch.log(probs[action_idx] + 1e-12)

            # 記録
            log_probs.append(log_prob)
            actions.append(action_idx)
            players.append(player)

            # 実行
            r, c = divmod(action_idx, BOARD_SIZE)
            board = make_move(board, (r, c), player)
            player *= -1

        # 対局終局時に報酬を決定（勝者:+1, 敗者:-1, 引き分け:0）
        black_count = np.sum(board == 1)
        white_count = np.sum(board == -1)
        if black_count > white_count:
            reward_for_black = 1.0
            reward_for_white = -1.0
        elif white_count > black_count:
            reward_for_black = -1.0
            reward_for_white = 1.0
        else:
            reward_for_black = 0.0
            reward_for_white = 0.0

        # 各手に対応する報酬（プレイヤーに応じて +1/-1/0）
        returns = []
        for p in players:
            returns.append(reward_for_black if p == 1 else reward_for_white)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # 基準（baseline）として平均を引いて分散を下げる（簡易）
        baseline = returns.mean() if len(returns) > 0 else 0.0
        advantages = returns - baseline

        # 損失 = - Σ (adv * log_prob)
        loss = -torch.sum(advantages * torch.stack(log_probs).to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # GUI 更新（あれば）
        if visualizer:
            visualizer.update(board)

        if (game + 1) % print_every == 0 or game == 0:
            print(f"Game {game+1} / {num_games}  |  loss: {loss.item():.4f}  |  black: {black_count} white: {white_count}")

    # モデル保存
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    return model


def evaluate_models(model_a, model_b, num_games=20, device="cpu"):
    """
    model_a (先手) と model_b (後手) を対戦させ、model_a 視点のスコアを返すリスト。
    1.0 = model_a の勝ち, 0.5 = 引き分け, 0.0 = model_a の負け
    """
    results = []

    for g in range(num_games):
        board = init_board()
        player = 1  # 先手は model_a
        models = {1: model_a, -1: model_b}

        while not is_game_over(board):
            moves = valid_moves(board, player)
            if not moves:
                player *= -1
                continue

            action = select_action(models[player], board, player, get_edge_index(), device=device, deterministic=True)
            if action is None:
                player *= -1
                continue

            board = make_move(board, action, player)
            player *= -1

        black_count = np.sum(board == 1)
        white_count = np.sum(board == -1)
        if black_count > white_count:
            # 先手（model_a）が黒なので、黒勝ち => model_a 勝ち
            results.append(1.0)
        elif white_count > black_count:
            results.append(0.0)
        else:
            results.append(0.5)

    return results


def train_loop(
    total_cycles=50,
    games_per_cycle=20,
    eval_games=20,
    print_every=1,
):
    """
    サイクルごとに学習 -> 評価 -> ELO 更新 を行う簡易ループ。
    """
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cycle in range(1, total_cycles + 1):
        print(f"===== Training cycle {cycle} =====")

        # 学習（自己対戦）
        current_model = train_gnn(num_games=games_per_cycle, print_every=print_every)

        # 現在モデルを評価対象（先手）として評価
        # baseline_model は以前の保存モデル（ここでは MODEL_PATH を上書きしているため、ロードして比較）
        baseline_model.load_state_dict(torch.load(MODEL_PATH))

        results = evaluate_models(current_model, baseline_model, num_games=eval_games, device=device)

        update_elo_from_match_results(elo, "current", "baseline", results)

        print(f"[Cycle {cycle}] ELO Ratings: {elo.ratings}")

        # 進捗保存
        torch.save(current_model.state_dict(), MODEL_PATH)
        elo.save("logs/elo_ratings.json")

    print("Training loop finished.")
