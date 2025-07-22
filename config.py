# config.py

# 게임 설정
BOARD_SIZE = 3

# MCTS 설정
NUM_SIMULATIONS = 100  # MCTS 시뮬레이션 횟수
CPUCT = 1.0     # 탐험(Exploration)과 활용(Exploitation)의 균형을 맞추는 상수
GENERATION_SIZE = 10  # MCTS에서 생성할 노드의 최대 수

# 훈련 설정
EPOCHS = 40
BATCH_SIZE = 64
LEARNING_RATE = 0.001
SELF_PLAY_GAMES = 400 # 한 세대의 훈련에 사용할 셀프 플레이 게임 수

# RAY
NUM_ACTOR = 8
RAY_RESULT_PATH = "ray_result"
CHECKPOINT_MODEL = "model.pth"

# 저장 경로
LATEST_MODEL = "model_latest.pth"