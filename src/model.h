#ifndef MODEL_H
#define MODEL_H

#if __has_include(<torch/torch.h>) 
    #include <torch/torch.h>
    #ifndef HAS_TORCH
        #define HAS_TORCH
    #endif
#endif

#include "board.h"

class DefaultEvaluation {
public:
    float operator()(Board& board, std::vector<Move>& legal_moves, std::vector<float>& move_weights);
    float operator()(Position* pos, std::vector<Move>& legal_moves, std::vector<float>& move_weights);
    template <Color Us> float operator()(Position* pos, MoveList<Us> legal_moves, std::vector<float>& move_weights);

private:
    float forward(Position* pos);
    float move_weight(Position* pos, Move m);
};


#ifdef HAS_TORCH

struct ModelConfig {
    int n_layers;       // Number of layers in the model
    int embed_dim;      // Embedding dimension
    int hidden_dim;     // Hidden dimension for intermediate layers
    int output_dim;     // Output dimension
};

struct Transformer : torch::nn::Module {
    Transformer(ModelConfig config);
    torch::Tensor forward( torch::Tensor );
private:
    ModelConfig config;
};


class TorchModel {
public:
    TorchModel(ModelConfig config);
    float operator()(Board& board, std::vector<Move>& legal_moves, std::vector<float>& move_weights);
    float operator()(Position* pos, std::vector<Move>& legal_moves, std::vector<float>& move_weights);
    template <Color Us> float operator()(Position* pos, MoveList<Us> legal_moves, std::vector<float>& move_weights);
private:
    Transformer model;
    ModelConfig config;
};

#endif

#endif 