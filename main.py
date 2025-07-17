# main.py

import ray

from trainer import train, collect_data
from play import play_game
from config import GENERATION_SIZE

def main():
    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    for _ in range(GENERATION_SIZE):
        all_training_data = collect_data()
        train(all_training_data)

    ray.shutdown()

    print("--- Starting Game ---")
    play_game()


if __name__ == "__main__":
    main()