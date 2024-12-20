import wrapper

# Create a board
board = wrapper.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

# Generate legal moves
legal_moves = board.get_legal_moves()

# Initialize DefaultEvaluation
eval = wrapper.DefaultEvaluation()
logits = [0.0] * len(legal_moves)  # Placeholder logits
evaluation_score = eval(board, legal_moves, logits)
print(f"Evaluation score: {evaluation_score}")

