# play.py

import torch
from game_logic import TicTacToe
from model import SimpleNN
from config import BOARD_SIZE, MODEL_PATH

def play_game():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 로드
    model = SimpleNN().to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device,weights_only=True))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Model file not found at {MODEL_PATH}. Playing with an untrained model.")
    model.eval()

    game = TicTacToe()
    human_player = -1 # 'O'

    while True:
        print("\n" + str(game))
        status = game.get_game_status()
        if status is not None:
            if status == 1: print("Player X wins!")
            elif status == -1: print("Player O wins!")
            else: print("It's a draw!")
            break

        if game.current_player == human_player:
            # 사람 턴
            valid_moves = game.get_valid_moves()
            while True:
                try:
                    move_str = input(f"Enter your move (row,col) for O: ")
                    row, col = map(int, move_str.split(','))
                    move = (row, col)
                    if move in valid_moves:
                        break
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid format. Use 'row,col'.")
        else:
            # AI 턴
            print("AI (X) is thinking...")
            board_tensor = torch.FloatTensor(game.get_board_state() * game.current_player).to(device)
            with torch.no_grad():
                policy, value_tensor = model(board_tensor)

            policy = torch.exp(policy).squeeze(0)
            mask = torch.zeros_like(policy)  # shape: (BOARD_SIZE * BOARD_SIZE,)

            legal_moves = game.get_valid_moves()  # [(i, j), ...]
            for i, j in legal_moves:
                idx = i * BOARD_SIZE + j
                mask[idx] = 1
            policy *= mask
            legal_moves_flatten = torch.tensor([i * BOARD_SIZE + j for i, j in legal_moves], device=device)
            print(policy)
            if policy.sum() == 0:
                move_idx = legal_moves_flatten[torch.randint(0, len(legal_moves_flatten), (1,))].item()
            else:
                policy = policy / policy.sum()  # normalize
                move_idx = torch.multinomial(policy, 1).item()

            move = (move_idx // BOARD_SIZE, move_idx % BOARD_SIZE)
            print(f"AI chose move: {move}")

        game.make_move(move)


if __name__ == "__main__":
    play_game()