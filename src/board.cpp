#include <sstream>
#include "board.h"

using namespace std;


Board::Board() {
    this->board = new Position;
    Position::set(DEFAULT_FEN, *(this->board));
}

Board::Board(string fen) {
    this->board = new Position;
    Position::set(fen, *(this->board));
}


void Board::play(Move move) {
    if (this->board->turn() == WHITE) {
        this->board->play<WHITE>(move);
    } else {
        this->board->play<BLACK>(move);
    }
}

void Board::undo(Move move) {
    if (this->board->ply() > 0) {
        if (this->board->turn() == WHITE) {
            this->board->undo<BLACK>(move);
        } else {
            this->board->undo<WHITE>(move);
        }
    }
}


vector<Move> Board::get_legal_moves() {
    vector<Move> m;
    if (this->board->turn() == WHITE) {
        MoveList<WHITE> move_list(*(this->board));
        
        for (int i = 0; i < move_list.size(); i++) {
            m.push_back(move_list[i]);
        }
    } else {
        MoveList<BLACK> move_list(*(this->board));
        for (int i = 0; i < move_list.size(); i++) {
            m.push_back(move_list[i]);
        }
    }
    return m;
}


string Board::to_string() {
    static thread_local string board_str;  // Ensure thread safety
    ostringstream oss;
    oss << *board;  // Use the operator<<
    board_str = oss.str();
    return board_str.c_str();
}

uint64_t Board::get_hash() {
    return this->board->get_hash();
}

Piece Board::piece_at(Square s) {
    return this->board->at(s);
}

Bitboard Board::piece_bitboard(Piece p) {
    return this->board->bitboard_of(p);
}

Color Board::turn() {
    return this->board->turn();
}

bool Board::is_white_turn() {
    return this->board->turn() == WHITE;
}

bool Board::is_repetition() {
    return this->board->is_repetition();
}

bool Board::is_rule_50() {
    return this->board->get_rule_50() == 100;
}

bool Board::is_insufficient() {
    if (this->board->bitboard_of(WHITE_PAWN) | this->board->bitboard_of(BLACK_PAWN) != 0)
        return false;
    if (this->board->bitboard_of(WHITE_ROOK) | this->board->bitboard_of(BLACK_ROOK) != 0) 
        return false;
    if (this->board->bitboard_of(WHITE_QUEEN) | this->board->bitboard_of(BLACK_QUEEN) != 0) 
        return false;


    int white_bishop = pop_count(this->board->bitboard_of(WHITE_BISHOP));
    int black_bishop = pop_count(this->board->bitboard_of(BLACK_BISHOP));
    
    if (white_bishop >= 2 || black_bishop >= 2) 
        return false;

    int white_knight = pop_count(this->board->bitboard_of(WHITE_KNIGHT));
    int black_knight = pop_count(this->board->bitboard_of(BLACK_KNIGHT));

    if (white_knight >= 3 || black_knight >= 3) 
        return false;
    if (white_bishop == 1 && white_knight >= 1) 
        return false;
    if (black_bishop == 1 && black_knight >= 1) 
        return false;

    return true;
}

Board::~Board() {
    delete this->board;
}

Position* Board::get_position() {
    return this->board;
}
