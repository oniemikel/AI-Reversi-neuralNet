import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QPushButton,
    QGridLayout,
    QLabel,
    QVBoxLayout,
)
from PyQt5.QtCore import Qt
from save_train_time import update_play_train_time
from reversi_game_logic import init_board, valid_moves, make_move, is_game_over
from reversi_cli import select_action, load_model, get_edge_index, board_to_tensor
from reversi_trainer import train_gnn_external_only

BOARD_SIZE = 8


class ReversiVisualizer:
    def __init__(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowTitle("Reversi Training Progress")
        self.central = QWidget()
        self.grid = QGridLayout()
        self.central.setLayout(self.grid)
        self.window.setCentralWidget(self.central)

        self.buttons = [
            [QLabel() for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)
        ]
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                label = self.buttons[r][c]
                label.setFixedSize(60, 60)
                label.setAlignment(Qt.AlignCenter)
                label.setStyleSheet(
                    """
                    background-color: #2e7d32;
                    border: 1px solid #1b5e20;
                    font-size: 40px;
                    font-weight: bold;
                    color: white;
                    """
                )
                self.grid.addWidget(label, r, c)

        self.window.setStyleSheet("background-color: #1b5e20;")
        self.window.show()

    def update(self, board):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = board[r][c]
                label = self.buttons[r][c]
                if val == 1:
                    label.setText("●")
                    label.setStyleSheet(
                        """
                        background-color: #2e7d32;
                        border: 1px solid #1b5e20;
                        font-size: 40px;
                        font-weight: bold;
                        color: black;
                        """
                    )
                elif val == -1:
                    label.setText("●")
                    label.setStyleSheet(
                        """
                        background-color: #2e7d32;
                        border: 1px solid #1b5e20;
                        font-size: 40px;
                        font-weight: bold;
                        color: white;
                        """
                    )
                else:
                    label.setText("")
                    label.setStyleSheet(
                        """
                        background-color: #2e7d32;
                        border: 1px solid #1b5e20;
                        """
                    )
        self.app.processEvents()


class ReversiGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reversi with PyQt")
        self.setStyleSheet("background-color: #1b5e20;")

        self.model = load_model()
        self.edge_index = get_edge_index()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.status_label = QLabel("Your turn (Black ●)")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            """
            font-size: 24px;
            font-weight: bold;
            color: white;
            padding: 10px;
            """
        )

        self.grid = QGridLayout()
        self.grid.setSpacing(1)

        self.buttons = [
            [QPushButton() for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)
        ]

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                btn = self.buttons[r][c]
                btn.setFixedSize(60, 60)
                btn.clicked.connect(self.make_move_closure(r, c))
                self.grid.addWidget(btn, r, c)

        # もう一度対戦ボタン
        self.restart_button = QPushButton("Play Again")
        self.restart_button.setStyleSheet(
            """
            background-color: #4caf50;
            color: white;
            font-size: 20px;
            padding: 8px;
            border-radius: 6px;
            """
        )
        self.restart_button.clicked.connect(self.reset_game)
        self.restart_button.hide()  # ゲーム終了時にのみ表示

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addLayout(self.grid)
        layout.addWidget(self.restart_button)
        self.central_widget.setLayout(layout)

        self.reset_game()

    def reset_game(self):
        self.board = init_board()
        self.player = 1
        self.history = []
        self.restart_button.hide()
        self.status_label.setText("Your turn (Black ●)")
        self.update_board()

    def make_move_closure(self, r, c):
        def handler():
            if self.player != 1:
                return
            if (r, c) not in valid_moves(self.board, self.player):
                self.status_label.setText("Invalid move! Try again.")
                return

            x = board_to_tensor(self.board, self.player)
            action_idx = r * 8 + c
            self.history.append((x, action_idx, self.player))

            self.board = make_move(self.board, (r, c), self.player)
            self.player *= -1
            self.update_board()

            if is_game_over(self.board):
                self.finish_game()
                return

            if not valid_moves(self.board, self.player):
                self.status_label.setText("AI passed. Your turn (Black ●)")
                self.player *= -1
                self.update_board()
                if not valid_moves(self.board, self.player):
                    self.finish_game()
                    return
            else:
                self.status_label.setText("AI's turn (White ○)")
                QApplication.processEvents()
                self.ai_move()

        return handler

    def ai_move(self):
        if is_game_over(self.board):
            self.finish_game()
            return

        moves = valid_moves(self.board, self.player)
        if not moves:
            self.status_label.setText("AI passed. Your turn (Black ●)")
            self.player *= -1
            self.update_board()
            return

        r, c = select_action(self.model, self.board, self.player, self.edge_index)
        self.board = make_move(self.board, (r, c), self.player)
        self.player *= -1
        self.update_board()

        if is_game_over(self.board):
            self.finish_game()
            return

        if not valid_moves(self.board, self.player):
            self.status_label.setText("You have no valid moves. Passing turn to AI.")
            self.player *= -1
            self.update_board()
            if valid_moves(self.board, self.player):
                QApplication.processEvents()
                self.ai_move()
        else:
            self.status_label.setText("Your turn (Black ●)")

    def update_board(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = self.board[r, c]
                btn = self.buttons[r][c]

                base_style = """
                    background-color: #2e7d32;
                    border: 1px solid #1b5e20;
                    font-size: 40px;
                    font-weight: bold;
                """

                if val == 1:
                    btn.setText("●")
                    btn.setStyleSheet(base_style + "color: black;")
                elif val == -1:
                    btn.setText("●")
                    btn.setStyleSheet(base_style + "color: white;")
                else:
                    btn.setText("")
                    btn.setStyleSheet(base_style)

                btn.setEnabled(
                    (self.player == 1)
                    and ((r, c) in valid_moves(self.board, self.player))
                )

    def finish_game(self):
        black_count = np.sum(self.board == 1)
        white_count = np.sum(self.board == -1)

        if black_count > white_count:
            result_msg = "Game Over. あなたの勝ち！"
        elif white_count > black_count:
            result_msg = "Game Over. AIの勝ち。"
        else:
            result_msg = "Game Over. 引き分け。"

        score_msg = f"""
            <div style='font-size:18px;'>
                <span style='color: black;'>●: {black_count}</span>
                &nbsp;
                <span style='color: white;'>●: {white_count}</span>
            </div>
        """

        self.status_label.setText(f"<b>{result_msg}</b><br>{score_msg}")

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                self.buttons[r][c].setEnabled(False)

        train_gnn_external_only(self.history)

        update_play_train_time(1)

        self.restart_button.show()


def main():
    app = QApplication(sys.argv)
    window = ReversiGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
