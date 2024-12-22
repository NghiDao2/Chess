#ifndef SPACE_H
#define SPACE_H

#include "../position/position.h"
#include "pawns.h"

const Bitboard space_mask[2] = {
    (MASK_FILE[CFILE] | MASK_FILE[DFILE] | MASK_FILE[EFILE] | MASK_FILE[FFILE]) & (MASK_RANK[RANK2] | MASK_RANK[RANK3] | MASK_RANK[RANK4]),
    (MASK_FILE[CFILE] | MASK_FILE[DFILE] | MASK_FILE[EFILE] | MASK_FILE[FFILE]) & (MASK_RANK[RANK5] | MASK_RANK[RANK6] | MASK_RANK[RANK7]),
};

template<Color Us>
inline int space_total(const Position* pos, Bitboard pawn_attack[2]) {

    Bitboard space_area = space_mask[Us] & (~pos->bitboard_of(Us, PAWN) & (~pawn_attack[~Us]));
    Bitboard additional_space = space_mask[Us] & (extend<relative_dir<Us>(SOUTH)>(pos->bitboard_of(Us, PAWN), 3));

    Bitboard blocked_mask = (shift<relative_dir<Us>(NORTH)>(pos->bitboard_of(Us, PAWN)) 
                        & (pawn_attack[~Us] | shift<relative_dir<Us>(SOUTH)>(pos->bitboard_of(~Us, PAWN))));

    int weight = pop_count(pos->all_pieces<Us>()) - 3 + std::min(pop_count(blocked_mask) * 2 , 9);

    return ((pop_count(space_area) + pop_count(additional_space)) * weight * weight)/16;
}


#endif 