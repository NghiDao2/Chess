#include "evaluation/evaluation.h"
#include "model.h"
#include <string>
#include <cmath>
#include <vector>


const float piece_weights[7] = {100, 300, 300, 500, 800, 12000, 0};

float DefaultEvaluation::move_weight(Position* pos, Move move) {
    int moved = pos->at(move.from()) & 0b0111;
    int target = pos->at(move.to()) & 0b0111;

    float weight = 0;
    if (move.is_capture()) {
        if (move.is_enpassant()) {
            weight += piece_weights[PAWN];
        }
        else {
            weight += piece_weights[target];
        }
    }

    if (move.is_promotion()) {
        Piece piece_promoted = move.piece_promotion();
        weight += piece_weights[piece_promoted];
    }

    if (moved != 0b111) {
        weight = -(10/(moved+1)); //slightly prioritizes moving stronger pieces
    }

    return std::tanh(weight/1200)/2 + 0.5;
}

float DefaultEvaluation::operator()(Board& board, std::vector<Move>& legal_moves, std::vector<float>& move_weights) {
    Position* pos = board.get_position();
    for (int i = 0; i < legal_moves.size(); i++) {
        move_weights[i] = this->move_weight(pos, legal_moves[i]);
    }
    return this->forward(pos);
}

float DefaultEvaluation::operator()(Position* pos, std::vector<Move>& legal_moves, std::vector<float>& move_weights) {
    for (int i = 0; i < legal_moves.size(); i++) {
        move_weights[i] = this->move_weight(pos, legal_moves[i]);
    }
    return this->forward(pos);
}

template<Color Us>
float DefaultEvaluation::operator()(Position* pos, MoveList<Us> legal_moves, std::vector<float>& move_weights) {
    for (int i = 0; i < legal_moves.size(); i++) {
        move_weights[i] = this->move_weight(pos, legal_moves[i]);
    }
    return this->forward(pos);
}

float DefaultEvaluation::forward(Position* pos) {
    Evaluation eval = Evaluation(pos);
    float x = float(eval.eval)/1200;
    return std::tanh(x);
}




#ifdef HAS_TORCH

Transformer::Transformer(ModelConfig config) {

}

TorchModel::TorchModel(ModelConfig config) : config(config), model(config) {
    this->config = config;
}

float TorchModel::operator()(Board& board, std::vector<Move>& legal_moves, std::vector<float>& move_weights) {
    return 0; //TO DO
}

float TorchModel::operator()(Position* pos, std::vector<Move>& legal_moves, std::vector<float>& move_weights) {
    return 0; //TO DO
}

template <Color Us> 
float TorchModel::operator()(Position* pos, MoveList<Us> legal_moves, std::vector<float>& move_weights) {
    return 0; //TO DO
}


#endif
