
#include "evaluation.h"
#include "pieces.h"
#include "mobility.h"
#include "data.h"
#include "space.h"

#include <vector>

using namespace std;


inline const int outpost_weights[5] = {0,22,36,23,36};
inline const int rook_file_weights[3] = {0,7,29};

int Evaluation::pieces_eg() {
    
    int v = 0;

    Bitboard outpost_mask[2];
    outpost_mask[WHITE] = pieces_outpost_mask<WHITE>(pawn_span, pawn_attack);
    outpost_mask[BLACK] = pieces_outpost_mask<BLACK>(pawn_span, pawn_attack);

    for (int i = WHITE_PAWN; i < WHITE_KING; i++) {
        Piece piece = Piece(i);
        PieceType piece_type = type_of(piece);
        Bitboard bitboard = pos->bitboard_of(piece);
        while (bitboard != 0) {
            Square sq = pop_lsb(&bitboard);
            v += outpost_weights[pieces_outpost_count<WHITE>(sq, piece_type, outpost_mask)];
            v -= 5 * bishop_xray_pawns<WHITE>(pos, sq, piece_type);
            v += 11 * rook_on_queen_file<WHITE>(pos, sq, piece_type);
            v += rook_file_weights[rook_on_file<WHITE>(pos, sq, piece_type)];
            v += 14 * queen_infiltration<WHITE>(pos, sq, piece_type, pawn_span[BLACK]);

            if (piece_type == BISHOP || piece_type == KNIGHT) 
                v -= (piece_type == KNIGHT ? 8 : 6) * distance_from_king<WHITE>(pos, sq);
            
        }
    } 
    
    for (int i = BLACK_PAWN; i < BLACK_KING; i++) {
        Piece piece = Piece(i);
        PieceType piece_type = type_of(piece);
        Bitboard bitboard = pos->bitboard_of(piece);
        while (bitboard != 0) {
            Square sq = pop_lsb(&bitboard);
            v += -outpost_weights[pieces_outpost_count<BLACK>(sq, piece_type, outpost_mask)];
            v -= -5 * bishop_xray_pawns<BLACK>(pos, sq, piece_type);
            v += -11 * rook_on_queen_file<BLACK>(pos, sq, piece_type);
            v += -rook_file_weights[rook_on_file<BLACK>(pos, sq, piece_type)];
            v += -14 * queen_infiltration<BLACK>(pos, sq, piece_type, pawn_span[WHITE]);

            if (piece_type == BISHOP || piece_type == KNIGHT) 
                v -= -(piece_type == KNIGHT ? 8 : 6) * distance_from_king<BLACK>(pos, sq);
        }
    } 

        
    //v += 16 * rook_on_king_ring(pos, square);
    //v += 24 * bishop_on_king_ring(pos, square);
    //v -= trapped_rook(pos, square) * 55 * (pos.c[0] || pos.c[1] ? 1 : 2);
    //v -= 56 * weak_queen(pos, square);
    
    //v += 45 * long_diagonal_bishop(pos, square);
    
    v += 3 * (pop_count(minor_behind_pawn_mask<WHITE>(pos)) - pop_count(minor_behind_pawn_mask<BLACK>(pos)));
    v -= 7 * (bishop_pawns<WHITE>(pos, pawn_attack[WHITE]) - bishop_pawns<BLACK>(pos, pawn_attack[BLACK]));

    return v;
}


int Evaluation::pawns_eg() {

    Bitboard weak_unopposed[2];
    weak_unopposed[WHITE] = pawn_backward[WHITE] | pawn_isolated[WHITE];
    weak_unopposed[BLACK] = pawn_backward[BLACK] | pawn_isolated[BLACK];

    int v = 0;

    v -= 56 * (pop_count(pawn_double_isolated[WHITE]) - pop_count(pawn_double_isolated[BLACK]));
    v -= 15 * (pop_count(pawn_isolated[WHITE] & ~pawn_double_isolated[WHITE])
             - pop_count(pawn_isolated[BLACK] & ~pawn_double_isolated[BLACK]));
    v -= 24 * (pop_count(pawn_backward[WHITE] & ~pawn_double_isolated[WHITE] & ~pawn_isolated[WHITE]) 
            - pop_count(pawn_backward[BLACK] & ~pawn_double_isolated[BLACK] & ~pawn_isolated[BLACK]));
    v -= 56 * (pop_count(pawn_doubled[WHITE]) - pop_count(pawn_doubled[BLACK]));
    v -= 27 * (pop_count(weak_unopposed[WHITE]) - pop_count(weak_unopposed[BLACK]));

    
    v -= 56 * (pop_count(pawn_weak_lever[WHITE]) - pop_count(pawn_weak_lever[BLACK]));
    
    v += -4 * (pop_count(pawn_blocked[WHITE] & MASK_RANK[RANK5]) - pop_count(pawn_blocked[BLACK] & MASK_RANK[RANK4]));
    v += 4 * (pop_count(pawn_blocked[WHITE] & MASK_RANK[RANK6]) - pop_count(pawn_blocked[BLACK] & MASK_RANK[RANK3]));
    
    v += pawn_connected_bonus<WHITE>(pos, false);
    v += pawn_connected_bonus<BLACK>(pos, false);

    return v;
}


int Evaluation::mobility_eg() {

    int v = 0;

    for (int i = WHITE_PAWN; i < WHITE_KING; i++) {
        Piece piece = Piece(i);
        PieceType piece_type = type_of(piece);
        Bitboard bitboard = pos->bitboard_of(piece);
        while (bitboard != 0) {
            Square sq = pop_lsb(&bitboard);
            v += mobility_bonus<WHITE>(pos, sq, piece_type, mobility_area[WHITE], false);
        }
    } 

    for (int i = BLACK_PAWN; i < BLACK_KING; i++) {
        Piece piece = Piece(i);
        PieceType piece_type = type_of(piece);
        Bitboard bitboard = pos->bitboard_of(piece);
        while (bitboard != 0) {
            Square sq = pop_lsb(&bitboard);
            v += mobility_bonus<BLACK>(pos, sq, piece_type, mobility_area[BLACK], false);
            
        }
    } 

    return v;
}



int Evaluation::end_game_eval() {
    
    int eval = 0;

    
    for (int i = 0; i < 14; i++) {
        if (i == WHITE_KING + 1) {i = BLACK_PAWN;}

        Piece piece = Piece(i);
        Color color = color_of(piece);
        PieceType piece_type = type_of(piece);
        Bitboard bitboard = pos->bitboard_of(piece);

        int count = 0;

        int sign = color ? -1 : 1;

        while (bitboard != 0) {
            Square sq = pop_lsb(&bitboard);
            eval += square_table_eg[piece_type][color ? h8 - sq : sq] * sign;
            count += 1;
        }

        int material = piece_value_eg[piece_type] * piece_count[piece];
        eval += material * sign;
    } 

    eval += pawns_eg();
    eval += pieces_eg();
    eval += mobility_eg();
    //
    //eval += threats_mg(pos);
    //eval += passed_mg(pos);


    //eval += king_mg(pos);
    //if (!nowinnable) eval += winnable_total_mg(pos, v);
    

    //std::cout << "pawns_eg " << pawns_eg() << std::endl;
    //std::cout << "pieces_eg " << pieces_eg() << std::endl;
    //std::cout << "mobility_eg " << mobility_eg() << std::endl;
    
    return eval;
}

