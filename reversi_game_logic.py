import numpy as np
import torch
from config import BOARD_SIZE


def init_board():
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    mid = BOARD_SIZE // 2
    board[mid - 1, mid - 1], board[mid, mid] = 1, 1
    board[mid - 1, mid], board[mid, mid - 1] = -1, -1
    return board


def valid_moves(board, player):
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    moves = []
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] != 0:
                continue
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                found_opponent = False
                while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    if board[nr, nc] == -player:
                        found_opponent = True
                    elif board[nr, nc] == player:
                        if found_opponent:
                            moves.append((r, c))
                        break
                    else:
                        break
                    nr += dr
                    nc += dc
    return list(set(moves))


def make_move(board, move, player):
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    r, c = move
    board = board.copy()
    board[r, c] = player
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        flip_positions = []
        while 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
            if board[nr, nc] == -player:
                flip_positions.append((nr, nc))
            elif board[nr, nc] == player:
                for fr, fc in flip_positions:
                    board[fr, fc] = player
                break
            else:
                break
            nr += dr
            nc += dc
    return board


def is_game_over(board):
    return not valid_moves(board, 1) and not valid_moves(board, -1)


def get_edge_index():
    edges = []
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            idx = r * BOARD_SIZE + c
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                    nidx = nr * BOARD_SIZE + nc
                    edges.append([idx, nidx])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index
