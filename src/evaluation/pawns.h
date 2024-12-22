#ifndef PAWNS_H
#define PAWNS_H

#include "../surge/src/position.h"

inline const Bitboard isolated_mask[8] = {
    MASK_FILE[1], 
    MASK_FILE[0] | MASK_FILE[2],
    MASK_FILE[1] | MASK_FILE[3],
    MASK_FILE[2] | MASK_FILE[4],
    MASK_FILE[3] | MASK_FILE[5],
    MASK_FILE[4] | MASK_FILE[6],
    MASK_FILE[5] | MASK_FILE[7],
    MASK_FILE[6]
};


template<Color Us>
inline Bitboard isolated_pawn_mask(const Position* pos) {
    Bitboard pawn = pos->bitboard_of(Us, PAWN);
    
    Bitboard b = 0;

    b |= (pawn & MASK_FILE[AFILE] & Bitboard((pawn & isolated_mask[AFILE]) == 0));
    b |= (pawn & MASK_FILE[BFILE] & Bitboard((pawn & isolated_mask[BFILE]) == 0));
    b |= (pawn & MASK_FILE[CFILE] & Bitboard((pawn & isolated_mask[CFILE]) == 0));
    b |= (pawn & MASK_FILE[DFILE] & Bitboard((pawn & isolated_mask[DFILE]) == 0));
    b |= (pawn & MASK_FILE[EFILE] & Bitboard((pawn & isolated_mask[EFILE]) == 0));
    b |= (pawn & MASK_FILE[FFILE] & Bitboard((pawn & isolated_mask[FFILE]) == 0));
    b |= (pawn & MASK_FILE[GFILE] & Bitboard((pawn & isolated_mask[GFILE]) == 0));
    b |= (pawn & MASK_FILE[HFILE] & Bitboard((pawn & isolated_mask[HFILE]) == 0));

    return b;
}

template<Direction dir>
inline Bitboard extend(Bitboard bitboard, int length) {
    Bitboard result = 0;
    for (int i = 0; i < length; i++) {
        bitboard = shift<dir>(bitboard);
        result |= bitboard;
    }
    return result;
}

template<Color Us>
inline Bitboard stacked_pawn_mask(const Position* pos) { // multiple pawns on the same file
        
    Bitboard pawn = pos->bitboard_of(Us, PAWN);
    return pawn & extend<relative_dir<Us>(NORTH)>(pawn, 6);
}

template<Color Us>
inline Bitboard doubled_pawn_mask(const Position* pos) {
    Bitboard pawn = pos->bitboard_of(Us, PAWN);
    return pawn & shift<relative_dir<Us>(NORTH)>(pawn);
}

template<Color Us>
inline Bitboard doubled_isolated_pawn_mask(const Position* pos, Bitboard pawn_isolated[2]) {
    return pawn_isolated[Us] & stacked_pawn_mask<Us>(pos) & extend<relative_dir<Us>(SOUTH)>(pawn_isolated[~Us], 6);
}

template<Color Us>
inline Bitboard pawn_attacks_mask(const Position* pos) {
    Bitboard pawn = pos->bitboard_of(Us, PAWN);

    return shift<relative_dir<Us>(NORTH_WEST)>(pawn)
         | shift<relative_dir<Us>(NORTH_EAST)>(pawn);
}

template<Color Us>
inline Bitboard backward_pawn_mask(const Position* pos, Bitboard pawn_span[2], Bitboard pawn_attack[2]) {

    Bitboard our_pawn = pos->bitboard_of(Us, PAWN);
    Bitboard their_pawn = pos->bitboard_of(Us, PAWN);
    Bitboard unsupported = (our_pawn & ~pawn_span[Us]);
    return (unsupported & pawn_span[~Us]) | (unsupported & shift<relative_dir<Us>(SOUTH)>(their_pawn));
}

template<Color Us>
inline Bitboard supported_pawn_mask(const Position* pos) { // a pawn is supported if there is a pawn in adjancent or diagonally below
    return (pawn_attacks_mask<Us>(pos) & pos->bitboard_of(Us, PAWN));
}



template<Color Us>
inline Bitboard phalanx_pawn_mask(const Position* pos) {
    Bitboard pawn = pos->bitboard_of(Us, PAWN);
    return pawn & (shift<WEST>(pawn) | shift<EAST>(pawn));
}

template<Color Us>
inline Bitboard blocked_pawn_mask(const Position* pos) {
    return (pos->bitboard_of(Us, PAWN) 
    & (MASK_RANK[relative_rank<Us>(RANK5)] | MASK_RANK[relative_rank<Us>(RANK6)]) 
    & shift<relative_dir<Us>(SOUTH)>(pos->bitboard_of(~Us, PAWN)));
}

template<Color Us>
inline Bitboard opposed_pawn_mask(const Position* pos) {
    return extend<relative_dir<Us>(SOUTH)>(pos->bitboard_of(~Us, PAWN), 7) & pos->bitboard_of(Us, PAWN);
}

template<Color Us>
inline Bitboard weak_lever_mask(const Position* pos, Bitboard pawn_attack[2]) {
    return (pos->bitboard_of(Us, PAWN) & shift<relative_dir<Us>(SOUTH_WEST)>(pos->bitboard_of(~Us, PAWN)))
     & (pos->bitboard_of(Us, PAWN) & shift<relative_dir<Us>(SOUTH_EAST)>(pos->bitboard_of(~Us, PAWN)))
     & (~pawn_attack[Us]);
}




inline const int connected_seed[8] = {0, 7, 8, 12, 29, 48, 86};

template<Color Us>
inline int pawn_connected_bonus(const Position* pos, bool mg) {
    
    Bitboard supported = supported_pawn_mask<Us>(pos);
    Bitboard phalanx = phalanx_pawn_mask<Us>(pos);

    Bitboard connected = supported | phalanx;

    Bitboard opposed = opposed_pawn_mask<Us>(pos) & connected;
    
    int v = 0;
    for (int i = RANK2; i < RANK7; i++) {
        Bitboard mask = MASK_RANK[relative_rank<Us>(Rank(i))];
        int bonus = connected_seed[i] * (2 * pop_count(connected & mask) + pop_count(phalanx & mask) - pop_count(opposed & mask));
        bonus += pop_count(supported & mask) * 21;

        if (mg) v += bonus;
        else v += bonus * (i-3) / 4;
    }


    if (Us == WHITE) {
        return v;
    }
    return -v;
}


#endif 