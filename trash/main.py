from board_wrapper import Move, Board
from monte_carlo import MonteCarlo
import math
import time

starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

board = Board(starting_fen)

player_is_white = True


iterations = 50000
search_model = MonteCarlo()


t = time.time()
n = 0
def perft(depth):
    global n
    if depth == 0:
        n += 1
        return
    for m in board.get_legal_moves():
        board.play(m)
        perft(depth-1)
        board.undo(m)
perft(4)
total_elapsed = time.time() - t
print("total nodes:", n, "in", total_elapsed)




while True:
    print(board)
    legal_moves = board.get_legal_moves()
    print(board.evaluate())

    if board.is_repetition():
        print("Repetition")
        exit()

    if board.is_insufficient():
        print("Insufficient")
        exit()

    if board.get_winner():
        print("Game Winner")
        exit()

    if True:#(player_is_white and board.is_white_turn()) or ((not player_is_white) and not (board.is_white_turn())):

        played = False
        while not played:
            user_input = input().strip()

            if user_input == "exit":
                exit()
            else:  
                for m in legal_moves:
                    if str(m) == user_input:
                        board.play(m)
                        played = True
                        break
                
                if not played: 
                    print("invalid move")

    else:
        
        start_time = time.time()
        m = search_model.search(board)
        board.play(m)

        total_elapsed = time.time() - start_time
        total_iterations_per_second = iterations / total_elapsed
        print(f"Total Iterations: {iterations}, Time Elapsed: {total_elapsed:.2f} seconds")
        print(f"Average Iterations per Second: {total_iterations_per_second:.2f}")
        print("Move made:", m)
