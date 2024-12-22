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
    
    string to_string() const;
    uint64_t get_hash() const;

    Piece piece_at(Square s) const;
    Bitboard piece_bitboard(Piece p) const;

    bool is_white_turn() const;
    bool is_repetition() const;
    bool is_insufficient() const;
    bool is_rule_50() const;
    
    int get_rule_50() const;
    int get_repetition() const;


    Color turn() const;
    bool can_cstle_king(Color side) const;
    bool can_cstle_queen(Color side) const;
    Square enpassant_square() const;
    Position* get_position() const;

    ~Board();

private:
    Position* board;
};



#endif 