def print_board(board):
    """
    8x8リバーシ盤面をコンソールに見やすく表示する関数。
    1: 黒(●), -1: 白(○), 0: 空き(-)
    boardはnumpy配列やリストの2次元配列を想定。
    """
    print("  0 1 2 3 4 5 6 7")
    for r in range(8):
        row_str = f"{r} "
        for c in range(8):
            cell = board[r][c]
            if cell == 1:
                row_str += "● "
            elif cell == -1:
                row_str += "○ "
            else:
                row_str += "- "
        print(row_str)
