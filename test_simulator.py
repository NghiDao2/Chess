from wrapper import Move, Board, Piece, initialize_all  # Import classes from wrapper.so
from wrapper import MonteCarlo, MonteCarloConfig, DefaultEvaluation, Simulator, SimulatorConfig
import math

initialize_all()


default_eval = DefaultEvaluation()
config1 = MonteCarloConfig()
config1.exploration_scale = 1.06
config1.exploration_decay = 0.45

white_player = MonteCarlo(default_eval, config1)
black_player = MonteCarlo(default_eval)


config = SimulatorConfig(white_player, black_player)
config.move_time = 100
game = Simulator(config)
game.run(log=True)

total_elapsed = game.get_time_elapsed()


if game.is_white_win():
    print("White wins")
elif game.is_black_win():
    print("Black wins")
else:
    print("draw")
print(f"Total moves: {len(game.get_move_sequence())}, Time Elapsed: {total_elapsed:.2f} seconds")
print(f"Average seconds per move: {(total_elapsed/len(game.get_move_sequence())):.2f}")
print(f"Average iterations per second: {(game.get_total_iterations()/total_elapsed):.2f}")


for m in game.get_move_sequence():
    print(m)