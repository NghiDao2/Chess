from wrapper import Move, Board, MonteCarlo, Piece, DefaultEvaluation  # Import classes from wrapper.so
from wrapper import initialize_all
import time

initialize_all()

# Initialize the chess board
starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
board = Board(starting_fen)

# Define the bot using Monte Carlo and search parameters
default_eval = DefaultEvaluation()
monte_carlo = MonteCarlo(default_eval) 
search_time_ms = 2000  # Bot search time in milliseconds (3 seconds)

# Game settings
player_is_white = True  # Player chooses to play as white or black


def display_legal_moves(legal_moves):
    """Helper to display legal moves."""
    print("Legal moves:", [str(move) for move in legal_moves])


def is_white_win(board):
    return board.piece_bitboard(Piece.BLACK_KING) == 0

def is_black_win(board):
    return board.piece_bitboard(Piece.WHITE_KING) == 0


# Main game loop
while True:
    
    print(board)

    # Check for game termination
    if board.is_repetition():
        print("Game over: Repetition detected!")
        break
    if board.is_insufficient():
        print("Game over: Insufficient material!")
        break
    if board.is_rule_50():
        print("Game over: Insufficient material!")
        break
    if is_white_win(board) or is_black_win(board):
        print(f"Game over: {get_winner_text(board)}")
        break

    # Determine if it's the player's turn
    if (board.is_white_turn() and player_is_white) or (not board.is_white_turn() and not player_is_white):
        print("Your turn!")
        legal_moves = board.get_legal_moves()
        display_legal_moves(legal_moves)

        played = False
        while not played:
            user_input = input("Enter your move: ").strip()

            if user_input.lower() == "exit":
                print("Exiting game. Goodbye!")
                exit()

            for move in legal_moves:
                if str(move) == user_input:
                    board.play(move)
                    played = True
                    break

            if not played:
                print("Invalid move. Try again.")

    else:
        # Bot's turn
        print("Bot is thinking...")
        start_time = time.time()

        # Perform Monte Carlo search
        best_move = monte_carlo.search(board, search_time_ms)

        board.play(best_move)

        elapsed_time = time.time() - start_time
        num_it = monte_carlo.get_iterations_searched()

        print(f"Bot played: {best_move}")
        print(f"Search took {elapsed_time:.2f} seconds.")
        print(f"Number iterations searched: {num_it}.")
        print(f"Iterations per second: {(num_it/elapsed_time):.2f}")
