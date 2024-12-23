from wrapper import Move, Board, Piece, initialize_all  # Import classes from wrapper.so
from wrapper import MonteCarlo, MonteCarloConfig, DefaultEvaluation, SimulatorBatch, Simulator, SimulatorConfig


import time
import math
import random
import os
import sys


def create_next_run_folder(path='simulations'):
    os.makedirs(path, exist_ok=True)
    existing_runs = {int(name[4:]) for name in os.listdir(path)
                     if name.startswith('sim_') and name[4:].isdigit()}
    X = 1
    while X in existing_runs:
        X += 1
    new_folder_path = os.path.join(path, f'sim_{X}')
    os.makedirs(new_folder_path)
    return new_folder_path


def save_bot_info(path, filename, bot_config):
    os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, filename)

    with open(file_path, 'w') as file:
        file.write(f"Number of bots: {len(bot_config)}\n\n")
        for i, bot in enumerate(bot_config):
            file.write(f"Bot {i + 1}\n")
            file.write(f"exploration_scale: {bot.exploration_scale}\n")
            file.write(f"exploration_decay: {bot.exploration_decay}\n\n")


def sort_indices_desc(values):
    return [index for index, _ in sorted(enumerate(values), key=lambda x: x[1], reverse=True)]



# Initialize system
initialize_all()

default_eval = DefaultEvaluation()

best_scale = 1
best_decay = 0.25
scale_std = 0.5
decay_std = 0.1

NUM_TRIALS = 1
NUM_BOTS = 4
NUM_PROCESSES = 4
MAX_MOVE_TIME = 200
MIN_MOVE_TIME = 100


simulator = SimulatorBatch(num_processes=NUM_PROCESSES)
result_path = create_next_run_folder(path="simulations")


for trial_n in range(1, NUM_TRIALS + 1, 1):

    print(f"Starting trial {trial_n}")

    bot_config = [MonteCarloConfig() for _ in range(NUM_BOTS)]

    bot_config[0].exploration_scale = best_scale
    bot_config[0].exploration_decay = best_decay

    for i in range(1, NUM_BOTS):
        bot_config[i].exploration_scale = random.gauss(best_scale, scale_std)
        bot_config[i].exploration_decay = random.gauss(best_decay, decay_std)

    games_play = {}
    threads = []
    total_games = 0

    for white_id in range(NUM_BOTS):
        for black_id in range(NUM_BOTS):
            if white_id == black_id:
                continue

            white_bot = MonteCarlo(default_eval, bot_config[white_id])
            black_bot = MonteCarlo(default_eval, bot_config[black_id])
            move_time = random.randint(MIN_MOVE_TIME, MAX_MOVE_TIME)
            
            config = SimulatorConfig(white_bot, black_bot)
            config.move_time = move_time
            config.move_limit = 200
            game = Simulator(config)
            simulator.add(game)
            games_play[(white_id, black_id)] = [game, white_bot, black_bot]
            total_games += 1


    run_path = os.path.join(result_path, f'trial_{trial_n}')
    os.makedirs(run_path, exist_ok=True)
    save_bot_info(run_path, "bots.txt", bot_config)

    start_time = time.time()

    while simulator.get_queue_size() > 0:
        sys.stdout.write(f"\rGames played {total_games - simulator.get_queue_size()}/{total_games}")
        sys.stdout.flush()
        time.sleep(1)


    win_amount = [0] * NUM_BOTS
    draw_amount = [0] * NUM_BOTS
    lose_amount = [0] * NUM_BOTS

    for white_id in range(NUM_BOTS):
        for black_id in range(NUM_BOTS):
            if white_id == black_id:
                continue

            game = games_play[(white_id, black_id)][0]
            game.save(run_path, f"Bot_{white_id}_vs_Bot_{black_id}", f"Bot {white_id}", f"Bot {black_id}")

            if game.is_white_win() == 1:
                win_amount[white_id] += 1
                lose_amount[black_id] += 1
            elif game.is_black_win():
                win_amount[black_id] += 1
                lose_amount[white_id] += 1
            else:
                draw_amount[white_id] += 1
                draw_amount[black_id] += 1

    score = [win - lose for win, lose in zip(win_amount, lose_amount)]
    bot_ranking = sort_indices_desc(score)

    print()
    print(f"Trial {trial_n} in {(time.time() - start_time):.2f}s:")
    j = 0
    for bot_id in bot_ranking:
        j += 1
        output = f"{j}. Bot {bot_id}"
        output += f" | {bot_config[bot_id].exploration_scale:.2f} - {bot_config[bot_id].exploration_decay:.2f}"
        output += f" | {win_amount[bot_id]} wins"
        output += f" | {draw_amount[bot_id]} draws"
        output += f" | {lose_amount[bot_id]} losses"
        print(output)
    print()

    best_bot_id = bot_ranking[0]
    scale_std = max(abs(bot_config[best_bot_id].exploration_scale - best_scale), 0.1)
    decay_std = max(abs(bot_config[best_bot_id].exploration_decay - best_decay), 0.1)
    best_scale = bot_config[best_bot_id].exploration_scale
    best_decay = bot_config[best_bot_id].exploration_decay

simulator.exit_thread()