from wrapper import Move, Board, Piece
import time
import os

starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

def is_white_win(board):
    return board.piece_bitboard(Piece.BLACK_KING) == 0

def is_black_win(board):
    return board.piece_bitboard(Piece.WHITE_KING) == 0


class GameSimulator():

    def __init__(self, w, b, move_time=2000, move_limit=400):
        self.white_player = w
        self.black_player = b
        self.winner = 0
        self.move_sequence = []
        self.move_time = move_time
        self.board = Board(starting_fen) 
        self.total_iterations = 0
        self.start_time = 0
        self.time_elapsed = 0
        self.move_limit = move_limit
        
    def run(self, output=False):

        self.start_time = time.time()

        if output:
            print(self.board)

        while True:

            if (len(self.move_sequence) > self.move_limit):
                break  #if no result after move_limit moves, then assume a draw
            if self.board.is_repetition():
                break
            if self.board.is_insufficient():
                break
            if len(self.board.get_legal_moves()) == 0:
                break
            if is_white_win(self.board):
                self.winner = 1
                break
            if is_black_win(self.board):
                self.winner = -1
                break

            iterations = 0
            if (self.board.is_white_turn()):
                m = self.white_player.search(self.board, self.move_time)
                self.board.play(m)
                self.move_sequence.append(m)
                iterations = self.white_player.get_iterations_searched()
            else:
                m = self.black_player.search(self.board, self.move_time)
                self.board.play(m)
                self.move_sequence.append(m)
                iterations = self.white_player.get_iterations_searched()
            
            self.total_iterations += iterations

            if output:
                print(self.board)
                print("Searched", iterations, "iterations")
                print("Move: ", self.move_sequence[-1])

        self.time_elapsed = time.time() - self.start_time


    def save(self, path, filename, white_name="Bot", black_name="Bot"):
        os.makedirs(path, exist_ok=True)
        file_path = os.path.join(path, f"{filename}.txt")

        with open(file_path, 'w') as file:
            file.write(f'[White "{white_name}"]\n')
            file.write(f'[Black "{black_name}"]\n')
            file.write(f'[Time per move {self.move_time}"]\n')
            file.write(f'[Time elapsed {self.time_elapsed}"]\n')
            file.write(f'[Total iterations {self.total_iterations}"]\n')

            if self.winner == 1:
                file.write(f'[Result 1-0]\n')
            elif self.winner == -1:
                file.write(f'[Result 0-1]\n')
            else:
                file.write(f'[Result 0-0]\n')

            file.write('\n')
            for i, move in enumerate(self.move_sequence):
                if i % 2 == 0:
                    file.write(f"{i}. {str(move)} ")
                else:
                    file.write(f"{str(move)} ")

            if self.winner == 1:
                file.write('1-0\n')
            elif self.winner == -1:
                file.write('0-1\n')
            else:
                file.write('0-0\n')