
#include "evaluation.h"
#include "pieces.h"
#include "mobility.h"
#include "data.h"
#include "space.h"

#include <vector>

using namespace std;



inline const vector<vector<int>> qo = {
    {38},
    {255, -62},
    {104, 4, 0},
    {-2, 47, 105, -208},
    {24, 117, 133, -134, -6},
};

inline const vector<vector<int>> qt = {
    {0},
    {63, 0},
    {65, 42, 0},
    {39, 24, -24, 0},
    {100, -42, 137, 268, 0}
};


int Evaluation::imbalance_total(int piece_count[]) {

    int v = 0;

    //Pawn
    v += qo[PAWN][PAWN] * piece_count[WHITE_PAWN]*piece_count[WHITE_PAWN];
    v -= qo[PAWN][PAWN] * piece_count[BLACK_PAWN]*piece_count[BLACK_PAWN];
    

    //Knight
    v += qo[KNIGHT][KNIGHT] * piece_count[WHITE_KNIGHT]*piece_count[WHITE_KNIGHT];
    v += qo[KNIGHT][PAWN] * piece_count[WHITE_KNIGHT]*piece_count[WHITE_PAWN];
    
    v -= qo[KNIGHT][KNIGHT] * piece_count[BLACK_KNIGHT]*piece_count[BLACK_KNIGHT];
    v -= qo[KNIGHT][PAWN] * piece_count[BLACK_KNIGHT]*piece_count[BLACK_PAWN];


    //Bishop
    v += qo[BISHOP][KNIGHT] * piece_count[WHITE_BISHOP]*piece_count[WHITE_KNIGHT];
    v += qo[BISHOP][PAWN] * piece_count[WHITE_BISHOP]*piece_count[WHITE_PAWN];
    
    v -= qo[BISHOP][KNIGHT] * piece_count[BLACK_BISHOP]*piece_count[BLACK_KNIGHT];
    v -= qo[BISHOP][PAWN] * piece_count[BLACK_BISHOP]*piece_count[BLACK_PAWN];


    //Rook
    v += qo[ROOK][ROOK] * piece_count[WHITE_ROOK]*piece_count[WHITE_ROOK];
    v += qo[ROOK][KNIGHT] * piece_count[WHITE_ROOK]*piece_count[WHITE_KNIGHT];
    v += qo[ROOK][PAWN] * piece_count[WHITE_ROOK]*piece_count[WHITE_PAWN];
    
    v -= qo[ROOK][ROOK] * piece_count[BLACK_ROOK]*piece_count[BLACK_ROOK];
    v -= qo[ROOK][KNIGHT] * piece_count[BLACK_ROOK]*piece_count[BLACK_KNIGHT];
    v -= qo[ROOK][PAWN] * piece_count[BLACK_ROOK]*piece_count[BLACK_PAWN];


    //Queen
    v += qo[QUEEN][QUEEN] * piece_count[WHITE_QUEEN]*piece_count[WHITE_QUEEN];
    v += qo[QUEEN][ROOK] * piece_count[WHITE_QUEEN]*piece_count[WHITE_ROOK];
    v += qo[QUEEN][KNIGHT] * piece_count[WHITE_QUEEN]*piece_count[WHITE_KNIGHT];
    v += qo[QUEEN][PAWN] * piece_count[WHITE_QUEEN]*piece_count[WHITE_PAWN];
    
    v -= qo[QUEEN][QUEEN] * piece_count[BLACK_QUEEN]*piece_count[BLACK_QUEEN];
    v -= qo[QUEEN][ROOK] * piece_count[BLACK_QUEEN]*piece_count[BLACK_ROOK];
    v -= qo[QUEEN][KNIGHT] * piece_count[BLACK_QUEEN]*piece_count[BLACK_KNIGHT];
    v -= qo[QUEEN][PAWN] * piece_count[BLACK_QUEEN]*piece_count[BLACK_PAWN];


    int w = piece_count[WHITE_BISHOP] + piece_count[WHITE_ROOK] + piece_count[WHITE_QUEEN];
    int b = piece_count[BLACK_BISHOP] + piece_count[BLACK_ROOK] + piece_count[BLACK_QUEEN];

    if (piece_count[WHITE_BISHOP] > 1) {
        v += -26 * w;
        v -= 46 * b;
        v += 1438;
    }
    if (piece_count[BLACK_BISHOP] > 1) {
        v += 46 * w;
        v -= -26 * b;
        v -= 1438;
    }

    return v/16;
}

inline const int outpost_weights[5] = {0,31,-7,30,56};
inline const int rook_file_weights[3] = {0,19,48};

int Evaluation::pieces_mg() {
    
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
            v -= 4 * bishop_xray_pawns<WHITE>(pos, sq, piece_type);
            v += 6 * rook_on_queen_file<WHITE>(pos, sq, piece_type);
            v += rook_file_weights[rook_on_file<WHITE>(pos, sq, piece_type)];
            v -= 2 * queen_infiltration<WHITE>(pos, sq, piece_type, pawn_span[BLACK]);

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
            v -= -4 * bishop_xray_pawns<BLACK>(pos, sq, piece_type);
            v += -6 * rook_on_queen_file<BLACK>(pos, sq, piece_type);
            v += -rook_file_weights[rook_on_file<BLACK>(pos, sq, piece_type)];
            v -= -2 * queen_infiltration<BLACK>(pos, sq, piece_type, pawn_span[WHITE]);

            if (piece_type == BISHOP || piece_type == KNIGHT) 
                v -= -(piece_type == KNIGHT ? 8 : 6) * distance_from_king<BLACK>(pos, sq);
        }
    } 

        
    //v += 16 * rook_on_king_ring(pos, square);
    //v += 24 * bishop_on_king_ring(pos, square);
    //v -= trapped_rook(pos, square) * 55 * (pos.c[0] || pos.c[1] ? 1 : 2);
    //v -= 56 * weak_queen(pos, square);
    
    //v += 45 * long_diagonal_bishop(pos, square);
    
    v += 18 * (pop_count(minor_behind_pawn_mask<WHITE>(pos)) - pop_count(minor_behind_pawn_mask<BLACK>(pos)));
    v -= 3 * (bishop_pawns<WHITE>(pos, pawn_attack[WHITE]) - bishop_pawns<BLACK>(pos, pawn_attack[BLACK]));

    return v;
}


int Evaluation::pawns_mg() {

    Bitboard weak_unopposed[2];
    weak_unopposed[WHITE] = pawn_backward[WHITE] | pawn_isolated[WHITE];
    weak_unopposed[BLACK] = pawn_backward[BLACK] | pawn_isolated[BLACK];

    int v = 0;

    v -= 11 * (pop_count(pawn_double_isolated[WHITE]) - pop_count(pawn_double_isolated[BLACK]));
    v -= 5 * (pop_count(pawn_isolated[WHITE] & ~pawn_double_isolated[WHITE])
             - pop_count(pawn_isolated[BLACK] & ~pawn_double_isolated[BLACK]));
    v -= 9 * (pop_count(pawn_backward[WHITE] & ~pawn_double_isolated[WHITE] & ~pawn_isolated[WHITE]) 
            - pop_count(pawn_backward[BLACK] & ~pawn_double_isolated[BLACK] & ~pawn_isolated[BLACK]));
    v -= 11 * (pop_count(pawn_doubled[WHITE]) - pop_count(pawn_doubled[BLACK]));
    v -= 13 * (pop_count(weak_unopposed[WHITE]) - pop_count(weak_unopposed[BLACK]));
    v += -11 * (pop_count(pawn_blocked[WHITE] & MASK_RANK[RANK5]) - pop_count(pawn_blocked[BLACK] & MASK_RANK[RANK4]));
    v += -3 * (pop_count(pawn_blocked[WHITE] & MASK_RANK[RANK6]) - pop_count(pawn_blocked[BLACK] & MASK_RANK[RANK3]));
    
    v += pawn_connected_bonus<WHITE>(pos, true);
    v += pawn_connected_bonus<BLACK>(pos, true);

    return v;
}


int Evaluation::mobility_mg() {

    int v = 0;

    for (int i = WHITE_PAWN; i < WHITE_KING; i++) {
        Piece piece = Piece(i);
        PieceType piece_type = type_of(piece);
        Bitboard bitboard = pos->bitboard_of(piece);
        
        while (bitboard != 0) {
            Square sq = pop_lsb(&bitboard);
            v += mobility_bonus<WHITE>(pos, sq, piece_type, mobility_area[WHITE], true);
        }
    } 

    for (int i = BLACK_PAWN; i < BLACK_KING; i++) {
        Piece piece = Piece(i);
        PieceType piece_type = type_of(piece);
        Bitboard bitboard = pos->bitboard_of(piece);
        while (bitboard != 0) {
            Square sq = pop_lsb(&bitboard);
            v += mobility_bonus<BLACK>(pos, sq, piece_type, mobility_area[BLACK], true);
            
        }
    } 

    return v;
}


int Evaluation::space() {
    
    if (material_mg[WHITE] - pawn_material_mg[WHITE] + material_mg[BLACK] - pawn_material_mg[BLACK] < 12222) {
        return 0;
    }
    return space_total<WHITE>(pos, pawn_attack) - space_total<BLACK>(pos, pawn_attack);
}


int Evaluation::middle_game_eval() {
    
    int eval = 0;

    int piece_count[NPIECES-1];
    material_mg[WHITE] = 0;
    material_mg[BLACK] = 0;
    pawn_material_mg[WHITE] = 0;
    pawn_material_mg[BLACK] = 0;
    
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
            eval += square_table_mg[piece_type][color ? h8 - sq : sq] * sign;
            count += 1;
        }

        piece_count[piece] = count;

        int material = piece_value_mg[piece_type] * piece_count[piece];
        eval += material * sign;
        if (piece_type != KING) {
            material_mg[color] += material;
            if (piece_type == PAWN) pawn_material_mg[color] += material;
        }

    } 
    eval += imbalance_total(piece_count);
    eval += pawns_mg();
    eval += pieces_mg();
    eval += mobility_mg();
    //
    //eval += threats_mg(pos);
    //eval += passed_mg(pos);

    eval += space();
    //eval += king_mg(pos);
    //if (!nowinnable) eval += winnable_total_mg(pos, v);
    

    //std::cout << "pawns_mg " << pawns_mg() << std::endl;
    //std::cout << "pieces_mg " << pieces_mg() << std::endl;
    //std::cout << "mobility_mg " << mobility_mg() << std::endl;
    //std::cout << "space " << space() << std::endl;
    
    return eval;
}

