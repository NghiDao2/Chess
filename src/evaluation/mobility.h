#ifndef MOBILITY_H
#define MOBILITY_H

#include "data.h"
#include "../position/position.h"



template<Color Us>
inline int mobility_bonus(const Position* pos, Square square, PieceType piece_type, Bitboard mobility_area, bool mg) {

    if (((pos->pinned >> square) & 0b1) == 1) {
        return 0;
    }

    int count = 0;
    Bitboard all_pieces = pos->all_pieces<Us>() | pos->all_pieces<~Us>();
    Bitboard our_queen = pos->bitboard_of(Us, QUEEN);
    Bitboard their_queen = pos->bitboard_of(~Us, QUEEN);
    Bitboard queen_mask = our_queen | their_queen;

    

    switch (piece_type)
    {
    case QUEEN:
        count = pop_count(attacks<QUEEN>(square, all_pieces) & mobility_area);
        break;
    case KNIGHT:
        count = pop_count(KNIGHT_ATTACKS[square] & mobility_area & ~our_queen);
        break;
    case BISHOP:
        count = pop_count(attacks<BISHOP>(square, all_pieces ^ queen_mask) & mobility_area & ~our_queen);
        break;
    case ROOK:
        count = pop_count(attacks<ROOK>(square, all_pieces ^ queen_mask ^ pos->bitboard_of(Us, ROOK)) & mobility_area);
        break;
    default:
        return 0;
    }
    
    int mobility;

    if (mg) {
        mobility = mobility_mg[piece_type][count];
    } else {
        mobility = mobility_eg[piece_type][count];
    }

    if (Us == WHITE) {
        return mobility;
    }
    return -mobility;
}

#endif 