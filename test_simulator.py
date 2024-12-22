from wrapper import Move, Board, Piece, initialize_all  # Import classes from wrapper.so
from wrapper import MonteCarlo, MonteCarloConfig, DefaultEvaluation
from game_simulator import GameSimulator
import math

initialize_all()


default_eval = DefaultEvaluation()
config1 = MonteCarloConfig()
config1.exploration_scale = 1.06
config1.exploration_decay = 0.45

white_player = MonteCarlo(default_eval, config1)
black_player = MonteCarlo(default_eval)


game = GameSimulator(white_player, black_player)
game.run(output=True)

total_elapsed = game.time_elapsed


print("Winner: ", game.winner)
print(f"Total moves: {len(game.move_sequence)}, Time Elapsed: {total_elapsed:.2f} seconds")
print(f"Average seconds per move: {(total_elapsed/len(game.move_sequence)):.2f}")
print(f"Average iterations per second: {(game.total_iterations/total_elapsed):.2f}")


for m in game.move_sequence:
    print(m)