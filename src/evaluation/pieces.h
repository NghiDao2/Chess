#ifndef PIECES_H
#define PIECES_H

#include "pawns.h"
#include "../surge/src/position.h"

template<Color C>
inline Bitboard pieces_outpost_mask(Bitboard pawn_span[2], Bitboard pawn_attack[2]) {
    
    Bitboard our_attack = pawn_attack[C];
    Bitboard their_span = pawn_span[~C];

    return our_attack 
    & MASK_RANK[relative_rank<C>(RANK4)] 
    & MASK_RANK[relative_rank<C>(RANK5)] 
    & MASK_RANK[relative_rank<C>(RANK6)] 
    & ~their_span;
}


template<Color Us>
inline Bitboard pieces_outpost_count(Square sq, PieceType type, Bitboard outpost_mask[2]) {
    
    if (type == KNIGHT) {
        if (((outpost_mask[Us] >> sq) & 0b1) == 1) {
            return 4;
        }
        return ((KNIGHT_ATTACKS[sq] & outpost_mask[Us]) != 0) ? 1 : 0;

    } else if (type == BISHOP) {
        return ((outpost_mask[Us] >> sq) & 0b1) == 1 ? 3 : 0;
    }

    return 0;
}

const Bitboard BLACK_SQUARE = 0xCCCCCCCCCCCCCCCC;
const Bitboard WHITE_SQUARE = 0x5555555555555555;

template<Color C>
inline Bitboard minor_behind_pawn_mask(Position* pos) {
    Bitboard pawn_behind;

    pawn_behind = shift<relative_dir<C>(SOUTH)>(pos->bitboard_of(C, PAWN));
    return pawn_behind & (pos->bitboard_of(C, KNIGHT) | pos->bitboard_of(C, BISHOP));
}

template<Color C>
inline int bishop_pawns(Position* pos, Bitboard pawn_attack) {
    Bitboard bishop_bitmap = pos->bitboard_of(C, BISHOP);
    Bitboard pawn_bitboard = pos->bitboard_of(C, PAWN);

    int v = 0;
    int white_pawns = pop_count(pawn_bitboard & WHITE_SQUARE);
    int black_pawns = pop_count(pawn_bitboard & BLACK_SQUARE);
    int white_bishops = pop_count(bishop_bitmap & WHITE_SQUARE);
    int black_bishops = pop_count(bishop_bitmap & BLACK_SQUARE);

    int blocked = shift<relative_dir<C>(NORTH)>(pawn_bitboard) & (pos->all_pieces<BLACK>() & pos->all_pieces<WHITE>());

    int white_unsupported = pop_count(bishop_bitmap & WHITE_SQUARE & ~pawn_attack);
    int black_unsupported = pop_count(bishop_bitmap & BLACK_SQUARE & ~pawn_attack);

    v += white_pawns * (white_bishops * blocked + white_unsupported);
    v += black_pawns * (black_bishops * blocked + black_unsupported);

    return v;
}


template<Color C>
inline int bishop_xray_pawns(Position* pos, Square sq, PieceType piece_type) {
    if (piece_type == BISHOP) {
        return pop_count(BISHOP_ATTACK_MASKS[sq] & pos->bitboard_of(~C, PAWN));
    } 
    return 0;
}

template<Color C>
inline int rook_on_queen_file(Position* pos, Square sq, PieceType piece_type) {
    if (piece_type == QUEEN) {
        return pop_count(MASK_FILE[file_of(sq)] & pos->bitboard_of(C, ROOK));
    } 
    return 0;
}

template<Color C>
inline int rook_on_file(Position* pos, Square sq, PieceType piece_type) {
    if (piece_type == ROOK) {
        if (pop_count(MASK_FILE[file_of(sq)] & pos->bitboard_of(C, PAWN)) > 0) {
            return 0;
        } else if (pop_count(MASK_FILE[file_of(sq)] & pos->bitboard_of(~C, PAWN)) > 0) {
            return 1;
        } 
        return 2;
    } 
    return 0;
}


template<Color C>
inline int queen_infiltration(Position* pos, Square sq, PieceType piece_type, Bitboard their_pawn_span) {
    if (piece_type == QUEEN && rank_of(sq) >= RANK5) {
        return ((their_pawn_span >> sq) & 0b1) == 0;
    } 
    return 0;
}

template<Color C>
inline int distance_from_king(Position* pos, Square sq) {
    Bitboard king_bitboard = pos->bitboard_of(C, KING);
    Square king = pop_lsb(&king_bitboard);

    int d_rank = std::abs(rank_of(sq) - rank_of(king));
    int d_file = std::abs(file_of(sq) - file_of(king));

    return std::max(d_rank, d_file) + std::min(d_rank, d_file);
}


#endif 