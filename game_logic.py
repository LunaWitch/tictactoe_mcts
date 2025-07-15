# game_logic.py

import numpy as np
from config import BOARD_SIZE

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
        self.current_player = 1  # 1: Player 1 (X), -1: Player 2 (O)

    def get_valid_moves(self):
        """ 유효한 수(빈 칸)의 위치를 반환 """
        return list(zip(*np.where(self.board == 0)))

    def make_move(self, move):
        """ 주어진 위치에 수를 놓고 플레이어를 변경 """
        row, col = move
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            self.current_player *= -1  # 플레이어 전환
            return True
        return False

    def get_game_status(self):
        """ 게임 상태를 확인: 승리, 무승부, 진행 중 """
        # 행/열/대각선 합산
        for i in range(BOARD_SIZE):
            if abs(self.board[i, :].sum()) == BOARD_SIZE:
                return self.board[i, 0]
            if abs(self.board[:, i].sum()) == BOARD_SIZE:
                return self.board[0, i]

        if abs(np.diag(self.board).sum()) == BOARD_SIZE:
            return self.board[0, 0]
        if abs(np.diag(np.fliplr(self.board)).sum()) == BOARD_SIZE:
            return self.board[0, -1]

        # 무승부 확인 (빈 칸 없음)
        if not self.get_valid_moves():
            return 0  # Draw

        return None  # Game is ongoing

    def get_board_state(self):
        """ 현재 보드 상태를 복사하여 반환 """
        return np.copy(self.board)

    def __str__(self):
        """ 보드를 문자열로 예쁘게 출력 """
        map_char = {1: 'X', -1: 'O', 0: '.'}
        return "\n".join([" ".join(map_char[cell] for cell in row) for row in self.board])
