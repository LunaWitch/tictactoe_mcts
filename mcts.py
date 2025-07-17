# mcts.py

import numpy as np
import math
import ray
import torch
from config import CPUCT, BOARD_SIZE, NUM_SIMULATIONS
from game_logic import TicTacToe
from util import load_latest_model

class Node:
    def __init__(self, parent, state, prior_p):
        self.parent = parent
        self.state = state
        self.children = {}  # {move: Node}
        self.N = 0  # 방문 횟수
        self.Q = 0  # 총 액션 가치
        self.P = prior_p  # 이 노드를 선택할 사전 확률 (모델로부터)

    def select(self):
        """ UCB 점수가 가장 높은 자식 노드를 선택 """
        return max(self.children.items(), key=lambda item: item[1].get_ucb_score())

    def expand(self, policy, game):
        """ 리프 노드를 확장 """
        valid_moves = game.get_valid_moves()
        for move in valid_moves:
            move_idx = move[0] * BOARD_SIZE + move[1]
            if move not in self.children:
                # 정책에서 해당 수의 확률을 가져옴
                self.children[move] = Node(self, game, policy[move_idx])

    def update(self, value):
        """ 노드 방문 횟수와 가치를 역전파하여 업데이트 """
        self.N += 1
        self.Q += value
        if self.parent:
            # 부모의 시점에서 가치를 업데이트해야 하므로 -value
            self.parent.update(-value)

    def get_ucb_score(self):
        """ Upper Confidence Bound (UCB) 점수 계산 """
        # Q/N (활용) + c_puct * P * sqrt(parent.N) / (1+N) (탐험)
        u = self.Q / self.N if self.N > 0 else 0
        c = CPUCT * self.P * math.sqrt(self.parent.N) / (1 + self.N)
        return u + c

@ray.remote(num_gpus=0.5)
class MCTS:
    def __init__(self):
        self.model, self.device = load_latest_model()
        
    def self_play(self):
        """ 셀프 플레이를 통해 학습 데이터 생성 """
        game_data = [] # (state, policy, value)
        
        game = TicTacToe()
        
        while True:
            # 현재 플레이어 시점의 보드 상태
            current_player_state = game.get_board_state() * game.current_player
            
            # MCTS를 통해 얻은 정책
            move_probs = self.search(game, NUM_SIMULATIONS)
            
            # 다음 수 선택 (가장 많이 방문한 노드)
            move_idx = np.random.choice(len(move_probs), p=move_probs)
            move = (move_idx // BOARD_SIZE, move_idx % BOARD_SIZE)

            # 데이터 저장: (상태, 정책)
            game_data.append([current_player_state, move_probs])
            
            game.make_move(move)
            status = game.get_game_status()

            if status is not None:
                # 게임 종료. 결과(value)를 모든 데이터에 할당
                for i in range(len(game_data)):
                    # (i-th player) * (winner) -> i-th player's perspective
                    value = status * ((-1)**(len(game_data) - 1 - i))
                    game_data[i].append(value)
                return game_data
            
    def search(self, game, simulations_number):
        """ MCTS 탐색을 수행하고 최적의 수를 찾음 """
        root_state = game.get_board_state() * game.current_player
        root = Node(None, root_state, 1.0)

        for _ in range(simulations_number):
            node = root
            temp_game = game.__class__()
            temp_game.board = np.copy(game.board)
            temp_game.current_player = game.current_player

            # 1. Selection
            while node.children:
                move, node = node.select()
                temp_game.make_move(move)

            # 2. Expansion & 3. Simulation(Value)
            status = temp_game.get_game_status()
            value = 0
            if status is None: # 게임이 끝나지 않았으면 확장
                board_tensor = torch.FloatTensor(temp_game.get_board_state() * temp_game.current_player).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    policy, value_tensor = self.model(board_tensor)

                policy = torch.exp(policy).squeeze(0).cpu().numpy()
                value = value_tensor.item()
                node.expand(policy, temp_game)
            else:
                value = status * temp_game.current_player

            # 4. Backpropagation
            node.update(-value)

        # 탐색 후 루트의 자식 노드 방문 횟수를 기반으로 정책 반환
        move_probs = np.zeros(BOARD_SIZE * BOARD_SIZE)
        for move, child in root.children.items():
            move_idx = move[0] * BOARD_SIZE + move[1]
            move_probs[move_idx] = child.N
        
        move_probs /= move_probs.sum()
        return move_probs