#ifndef EVALUATION_H
#define EVALUATION_H

#include "pawns.h"
#include "pieces.h"


class Evaluation {

public:
    Evaluation(Position* position) {
        pos = position;

        pawn_attack[WHITE] = pawn_attacks_mask<WHITE>(pos);
        pawn_attack[BLACK] = pawn_attacks_mask<BLACK>(pos);

        pawn_span[WHITE] = extend<NORTH>(pawn_attack[WHITE], 6) | pawn_attack[WHITE];
        pawn_span[BLACK] = extend<NORTH>(pawn_attack[BLACK], 6) | pawn_attack[BLACK];

        pawn_isolated[WHITE] = isolated_pawn_mask<WHITE>(pos);
        pawn_isolated[BLACK] = isolated_pawn_mask<BLACK>(pos);
        pawn_double_isolated[WHITE] = doubled_isolated_pawn_mask<WHITE>(pos, pawn_isolated);
        pawn_double_isolated[BLACK] = doubled_isolated_pawn_mask<BLACK>(pos, pawn_isolated);
        pawn_doubled[WHITE] = doubled_pawn_mask<WHITE>(pos);
        pawn_doubled[BLACK] = doubled_pawn_mask<BLACK>(pos);
        pawn_backward[WHITE] = backward_pawn_mask<WHITE>(pos, pawn_span, pawn_attack);
        pawn_backward[BLACK] = backward_pawn_mask<BLACK>(pos, pawn_span, pawn_attack);
        pawn_blocked[WHITE] = blocked_pawn_mask<WHITE>(pos);
        pawn_blocked[BLACK] = blocked_pawn_mask<BLACK>(pos);
        pawn_weak_lever[WHITE] = weak_lever_mask<WHITE>(pos, pawn_attack);
        pawn_weak_lever[BLACK] = weak_lever_mask<BLACK>(pos, pawn_attack);

        
        mobility_area[WHITE] = (~Bitboard(0)) 
        & (~pos->bitboard_of(WHITE_KING))
         & (~pos->bitboard_of(WHITE_QUEEN)) 
         & (~(pos->bitboard_of(WHITE_PAWN) & (MASK_RANK[RANK2] | MASK_RANK[RANK3])))
         & (~pawn_attack[BLACK]);

        mobility_area[BLACK] = (~Bitboard(0))
         & (~pos->bitboard_of(BLACK_KING)) 
         & (~pos->bitboard_of(BLACK_QUEEN)) 
         & (~(pos->bitboard_of(BLACK_PAWN) & (MASK_RANK[RANK7] | MASK_RANK[RANK6])))
         & (~pawn_attack[WHITE]);

    
        for (int i = 0; i < 14; i++) {
            if (i == WHITE_KING + 1) {i = BLACK_PAWN;}
            Piece piece = Piece(i);
            Color color = color_of(piece);
            Bitboard bitboard = pos->bitboard_of(piece);
            piece_count[piece] = pop_count(bitboard);
        } 
        
        int imbalance = imbalance_total(piece_count);
        
        int mg = this->middle_game_eval() + imbalance;
        int eg = this->end_game_eval() + imbalance;
        int p = this->phase();
        int r50 = pos->get_rule_50();
        
        eg = eg * this->scale_factor() / 64;
        this->eval = (((mg * p + ((eg * (128 - p)) << 0)) / 128) << 0);
        
        /*
        //if (arguments.length == 1) v = ((v / 16) << 0) * 16; //not quite sure what this is for

        */
        this->eval += (pos->turn() == WHITE) ? 28 : -28;
        this->eval = (this->eval * (100 - r50) / 100) << 0;
        //return v;

        //std::cout << "rule_50: " << r50 << std::endl;
        //std::cout << "phase: " << this->phase() << std::endl;
        //std::cout << "scale factor: " << this->scale_factor() << std::endl;
    }


    int eval;

private:
    Position* pos;

    Bitboard pawn_isolated[2];
    Bitboard pawn_double_isolated[2];
    Bitboard pawn_doubled[2];
    Bitboard pawn_backward[2];
    Bitboard pawn_blocked[2];
    Bitboard pawn_attack[2];
    Bitboard pawn_span[2];
    Bitboard pawn_weak_lever[2];
    Bitboard mobility_area[2];

    int material_mg[2];
    int pawn_material_mg[2];
    int piece_count[NPIECES-1];


    int middle_game_eval();
    int end_game_eval();
    int pawns_mg();
    int pawns_eg();
    int pieces_mg();
    int pieces_eg();
    int mobility_mg();
    int mobility_eg();
    int space();
    int phase();
    int scale_factor();
    int imbalance_total(int piece_count[]);
};



inline int Evaluation::phase() {

  int midgameLimit = 15258;
  int endgameLimit  = 3915;
  int npm = material_mg[WHITE] + material_mg[BLACK] - pawn_material_mg[WHITE] - pawn_material_mg[BLACK];
  npm = std::max(endgameLimit, std::min(npm, midgameLimit));
  return ((npm - endgameLimit) * 128) / (midgameLimit - endgameLimit);

}

inline int Evaluation::scale_factor() {
    
    int sf = 64; // Default scale factor (full evaluation)
    
    // Material and pawn counts
    int pc_w = piece_count[WHITE_PAWN];
    int pc_b = piece_count[BLACK_PAWN];
    int npm_w = material_mg[WHITE] - pawn_material_mg[WHITE];
    int npm_b = material_mg[BLACK] - pawn_material_mg[BLACK];



    bool opposite_bishop = (pos->bitboard_of(WHITE_BISHOP) & WHITE_SQUARE) != 0
                        && (pos->bitboard_of(WHITE_BISHOP) & BLACK_SQUARE) != 0
                        && (pos->bitboard_of(BLACK_BISHOP) & WHITE_SQUARE) != 0
                        && (pos->bitboard_of(BLACK_BISHOP) & BLACK_SQUARE) != 0;



    // 1. If white has no pawns and insufficient material
    if (pc_w == 0 && npm_w - npm_b <= 825) {
        sf = npm_w < 1276 ? 0 : npm_b <= 825 ? 4 : 14;
    }

    
    // 2. Opposite-colored bishops with minimal material
    else if (opposite_bishop) {
        if (npm_w == 825 && npm_b == 825) {
        sf = 30; // Slight reduction in evaluation
        } else {
        sf = 22; // General case for opposite bishops
        }
    }

    // 3. Simplified rook/pawn endgame condition
    else if (npm_w == 1276 && npm_b == 1276 && std::abs(pc_w - pc_b) <= 1) {
        sf = 36; // Scale down for rook endgames with few pawns
    }

    // 4. Minor adjustments for queen endgames or many pawns
    else {
        sf = std::min(sf, 36 + 5 * pc_w); // Adjust based on pawn count
    }

    return sf; // Return the scaled evaluation factor
}

#endif 