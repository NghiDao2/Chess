#ifndef MODEL_H
#define MODEL_H

#if __has_include(<torch/torch.h>) 
    #include <torch/torch.h>
    #ifndef HAS_TORCH
        #define HAS_TORCH
    #endif
#endif

#include "board.h"
#include <memory>
#include <pthread.h>
#include <queue>



class Model {
public:
    virtual float operator()(const Board& board, std::vector<Move>& legal_moves, std::vector<float>& move_weights) = 0; 
    virtual ~Model() = default;    
};

class DefaultEvaluation : public Model {
public:
    float operator()(const Board& board, std::vector<Move>& legal_moves, std::vector<float>& move_weights);

private:
    float forward(const Position* pos);
    float move_weight(const Position* pos, Move m);
};


#ifdef HAS_TORCH

// LayerNorm Struct
class LayerNorm : public torch::nn::Module {
public:
    LayerNorm(int64_t ndim, bool bias = false);
    torch::Tensor forward(const torch::Tensor& input);
private:
    torch::Tensor weight, bias;
};

// SelfAttention Struct
class SelfAttention : public torch::nn::Module {
public:
    SelfAttention(int64_t n_embed, int64_t n_head, float dropout = 0.0, bool bias = false);
    torch::Tensor forward(const torch::Tensor& x);
private:
    int64_t n_embed;
    int64_t n_head;
    float dropout;
    bool bias;
    torch::nn::Linear attention{nullptr}, projection{nullptr};
    torch::nn::Dropout residual_dropout{nullptr};
};

// MLP Struct
class MLP : public torch::nn::Module {
public:
    MLP(int64_t n_embed, float dropout = 0.0, bool bias = false);
    torch::Tensor forward(const torch::Tensor& x);
private:
    torch::nn::Linear up_sample{nullptr}, down_sample{nullptr};
    torch::nn::GELU gelu{nullptr};
    torch::nn::Dropout dropout_layer{nullptr};
};

// Block Struct
class Block : public torch::nn::Module {
public:
    Block(int64_t n_embed, int64_t n_head, float dropout = 0.0, bool bias = false);
    torch::Tensor forward(const torch::Tensor& x);
private:
    std::shared_ptr<LayerNorm> layer_norm_1, layer_norm_2;
    std::shared_ptr<SelfAttention> attention;
    std::shared_ptr<MLP> mlp;
};

// ChessModel Struct
class ChessModel : public torch::nn::Module {
public:
    ChessModel(int64_t n_layer, int64_t n_head, int64_t n_embed, float dropout = 0.0, bool bias = false);
    vector<torch::Tensor> forward(const torch::Tensor& x);

    /*
    //returns a tensor representing the input relative to the side making the move
    */
    torch::Tensor board_to_tensor(const Board& board);
    int64_t get_num_params() const;

private:
    int64_t n_layer;
    int64_t n_head;
    int64_t n_embed;
    float dropout;
    bool bias = false;

    torch::nn::Embedding piece_embed{nullptr};
    torch::nn::Embedding position_embed{nullptr};
    torch::nn::Embedding castling_embed{nullptr};
    torch::nn::Embedding enpassant_embed{nullptr};
    torch::nn::Embedding rule50_embed{nullptr};
    torch::nn::Embedding repetition_embed{nullptr};
    std::vector<std::shared_ptr<Block>> blocks;
    std::shared_ptr<LayerNorm> layer_norm;
    torch::nn::Linear policy{nullptr};
    torch::nn::Linear evaluation{nullptr};
};



class ModelConfig {
public:
    int64_t n_layer = 5;
    int64_t n_head = 8;
    int64_t n_embed = 256;
    float dropout = 0.0;
    bool bias = false; // True: bias in Linears and LayerNorms.
};

class ModelInput {
public:
    ModelInput(const torch::Tensor& i) : input(i), completed(false) {}

    const torch::Tensor input;
    torch::Tensor policy;
    torch::Tensor eval;
    bool completed;
};



class TorchModel : public Model {
public:
    TorchModel(ModelConfig config);
    float operator()(const Board& board, std::vector<Move>& legal_moves, std::vector<float>& move_weights);
    ~TorchModel();

    void set_evaluation_batch(int size);
    void to(const std::string& device_str);
    void eval_mode();
    void train_mode();
    
    std::vector<torch::Tensor> calculate_loss(vector<vector<Move>> games, int batch_size);

    friend void* torch_model_worker(void* arg);
    friend void accumulate_gradient(TorchModel main_model, std::vector<TorchModel> models);
    friend void synchronize_parameters(TorchModel main_model, std::vector<TorchModel> models);
private:

    uint16_t move_to_idx[16384];
    uint16_t idx_to_move[1882];

    int evaluation_batch;

    ModelConfig config;
    std::queue<ModelInput*> input_queue;
    ChessModel model;
    pthread_t worker_thread;
    pthread_mutex_t lock;
    pthread_cond_t input_added;
    pthread_cond_t finished_batch;

    torch::Device device;

    bool thread_exit = false;
};


#endif

#endif 