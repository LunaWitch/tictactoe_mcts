# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import BOARD_SIZE

class SimpleNN(nn.Module):
    """
    사용자 정의 간단한 완전 연결 신경망 모델.
    """
    def __init__(self, board_size=BOARD_SIZE, action_size=BOARD_SIZE*BOARD_SIZE):
        """
        모델의 레이어를 초기화합니다.

        :param board_size: 보드의 한 변의 크기 (예: 3 for 3x3)
        :param action_size: 가능한 모든 행동의 수 (예: 9 for 3x3)
        """
        super().__init__()
        self.board_size = board_size
        input_size = board_size * board_size

        # 공통 레이어
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)

        # 정책(Policy)을 위한 출력 레이어
        self.policy_head = nn.Linear(64, action_size)
        
        # 가치(Value)를 위한 출력 레이어
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        """
        신경망의 순전파를 정의합니다.

        :param x: 입력 텐서 (보드 상태)
        :return: (정책, 가치) 튜플
        """
        # 입력 데이터를 1차원으로 펼침
        x = x.view(-1, self.board_size * self.board_size)
        
        # 공통 레이어 통과
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # 정책 헤드
        # 결과는 각 행동에 대한 로그 확률
        policy = F.log_softmax(self.policy_head(x), dim=1)

        # 가치 헤드
        # 결과는 -1과 1 사이의 값으로, 현재 상태의 승리 확률을 예측
        value = torch.tanh(self.value_head(x))

        return policy, value