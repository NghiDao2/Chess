#ifndef BOARD_H
#define BOARD_H

#include "surge/src/position.h" 
#include "surge/src/tables.h"
#include "surge/src/types.h"
#include <cstdint> 
#include <vector> 


using namespace std;


class Board {
public:
    Board();
    Board(string fen);

    void play(Move move);
    void undo(Move move);

    vector<Move> get_legal_moves();
    
    string to_string();
    uint64_t get_hash();

    Piece piece_at(Square s);
    Bitboard piece_bitboard(Piece p);

    Color turn();
    bool is_white_turn();
    bool is_repetition();
    bool is_insufficient();
    bool is_rule_50();

    Position* get_position();

    ~Board();

private:
    Position* board;
};



#endif 