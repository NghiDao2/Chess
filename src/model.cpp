#include "evaluation/evaluation.h"
#include "model.h"
#include "white_moves.h"
#include <string>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <algorithm>


const float piece_weights[7] = {100, 300, 300, 500, 800, 12000, 0};

float DefaultEvaluation::move_weight(const Position* pos, Move move) {
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

float DefaultEvaluation::operator()(const Board& board, std::vector<Move>& legal_moves, std::vector<float>& move_weights) {
    const Position* pos = board.get_position();
    for (int i = 0; i < legal_moves.size(); i++) {
        move_weights[i] = this->move_weight(pos, legal_moves[i]);
    }
    return this->forward(pos);
}

float DefaultEvaluation::forward(const Position* pos) {
    Evaluation eval = Evaluation(pos);
    float x = float(eval.eval)/1200;
    return std::tanh(x);
}




///////////////////////////////////
///////////////////////////////////




#ifdef HAS_TORCH


void* torch_model_worker(void* arg) {
    TorchModel* model = static_cast<TorchModel*>(arg);

    while (true) {
    
        pthread_mutex_lock(&model->lock);

        while (model->input_queue.empty() && model->thread_exit == false) {
            pthread_cond_wait(&model->input_added, &model->lock);
        }


        if (model->thread_exit == true) {  
            pthread_mutex_unlock(&model->lock);
            break;
        }

        std::vector<ModelInput*> object_batch;
        std::vector<torch::Tensor> batch;

        for (int i = 0; i < model->evaluation_batch; i++) {
            if (model->input_queue.empty()) {
                break;
            }

            ModelInput* input = model->input_queue.front();
            object_batch.push_back(input);   
            batch.push_back(input->input);    
            model->input_queue.pop();
        }
        
        pthread_mutex_unlock(&model->lock);

        torch::Tensor batch_tensor = torch::stack(batch).to(model->device);
        std::vector<torch::Tensor> output = model->model.forward(batch_tensor);
        


        pthread_mutex_lock(&model->lock);

        for (int i = 0; i < object_batch.size(); i++) {
            object_batch[i]->completed = true;
            object_batch[i]->eval = output[0][i];
            object_batch[i]->policy = output[1][i];
        }

        pthread_cond_broadcast(&model->finished_batch);

        pthread_mutex_unlock(&model->lock);

    }


    return nullptr;
}


void TorchModel::set_evaluation_batch(int size) {    
    pthread_mutex_lock(&this->lock);
    this->evaluation_batch = size;
    pthread_mutex_unlock(&this->lock);
}

TorchModel::TorchModel(ModelConfig conf) : 
config(conf), 
evaluation_batch(5000),
device(torch::kCPU),
model(
    conf.n_layer,
    conf.n_head, 
    conf.n_embed, 
    conf.dropout,
    conf.bias
) {

    pthread_mutex_init(&this->lock, nullptr);
    pthread_cond_init(&this->input_added, nullptr);
    pthread_cond_init(&this->finished_batch, nullptr);

    int result = pthread_create(&(this->worker_thread), NULL, &torch_model_worker, this);
    if (result != 0) {
        std::cerr << "Error: pthread_create failed" << std::endl;
        exit(1);
    }

    this->model.to(this->device);
}


float TorchModel::operator()(const Board& board, std::vector<Move>& legal_moves, std::vector<float>& move_weights) {
    
    torch::Tensor input_tensor = this->model.board_to_tensor(board);
    ModelInput input(input_tensor);

    pthread_mutex_lock(&this->lock);
    this->input_queue.push(&input);
    

    pthread_cond_signal(&this->input_added);

    while (input.completed == false) {
        pthread_cond_wait(&this->finished_batch, &this->lock);
    }

    pthread_mutex_unlock(&this->lock);

    
    for (int i = 0; i < legal_moves.size(); i++) {
        std::string move_string = legal_moves[i].to_string();

        auto it = WHITE_MOVE_TO_IDX.find(move_string);

        if (it == WHITE_MOVE_TO_IDX.end()) {
            std::cerr << "Move " << move_string << " was not found in white_moves.h" << std::endl;
        }

        int move_idx = it->second;
        move_weights[i] = input.policy[move_idx].item<float>();
    }

    return input.eval.item<float>();
}


std::vector<torch::Tensor> TorchModel::calculate_loss(
        std::vector<std::vector<Move>> white_wins, 
        std::vector<std::vector<Move>> black_wins,
        int batch_size) 
{

//at::autocast_mode::AutocastMode autocast(this->device, torch::kBFloat16);

    vector<torch::Tensor> inputs;
    vector<torch::Tensor> expected_evals;
    vector<torch::Tensor> expected_logits;

}


TorchModel::~TorchModel() {
    this->thread_exit = true;
    pthread_cond_broadcast(&this->input_added);
    pthread_join(this->worker_thread, nullptr);

    pthread_mutex_destroy(&this->lock);
    pthread_cond_destroy(&this->input_added);
    pthread_cond_destroy(&this->finished_batch);
}


void TorchModel::to(const std::string& device_str) {
// Convert string to torch::Device
    this->device = torch::Device(device_str);
    this->model.to(this->device);

}

void TorchModel::eval_mode() {
    model.eval();
}

void TorchModel::train_mode() {
    model.train();
}




///////////////////////////////////////////
///////////////////////////////////////////


// LayerNorm Implementation
LayerNorm::LayerNorm(int64_t ndim, bool bias) {
    this->weight = register_parameter("weight", torch::ones({ndim}));
    if (bias) {
        this->bias = register_parameter("bias", torch::zeros({ndim}));
    }
}

torch::Tensor LayerNorm::forward(const torch::Tensor& input) {
    // Normalize only the last dimension (features)
    return torch::layer_norm(input, {input.size(-1)}, this->weight, this->bias, 1e-5);
}

// SelfAttention Implementation
SelfAttention::SelfAttention(int64_t n_embed, int64_t n_head, float dropout, bool bias)
    : n_embed(n_embed), n_head(n_head), dropout(dropout), bias(bias) {
    attention = register_module("attention", torch::nn::Linear(torch::nn::LinearOptions(n_embed, 3 * n_embed).bias(bias)));
    projection = register_module("projection", torch::nn::Linear(torch::nn::LinearOptions(n_embed, n_embed).bias(bias)));
    residual_dropout = register_module("residual_dropout", torch::nn::Dropout(dropout));
}

torch::Tensor SelfAttention::forward(const torch::Tensor& x) {

    std::vector<int64_t> sizes = x.sizes().vec();
    int64_t B = sizes[0];
    int64_t T = sizes[1];
    int64_t C = sizes[2];

    std::vector<torch::Tensor> qkv = this->attention->forward(x).chunk(3, -1);

    torch::Tensor q = qkv[0].contiguous().view({B, T, this->n_head, C / this->n_head}).transpose(1, 2);
    torch::Tensor k = qkv[1].contiguous().view({B, T, this->n_head, C / this->n_head}).transpose(1, 2);
    torch::Tensor v = qkv[2].contiguous().view({B, T, this->n_head, C / this->n_head}).transpose(1, 2);

    torch::Tensor output = torch::scaled_dot_product_attention(q, k, v, {}, dropout);

    output = output.transpose(1, 2).contiguous().view({B, T, C});

    return this->residual_dropout->forward(this->projection->forward(output));
}



// MLP Implementation
MLP::MLP(int64_t n_embed, float dropout, bool bias) {
    up_sample = register_module("up_sample", torch::nn::Linear(torch::nn::LinearOptions(n_embed, 4 * n_embed).bias(bias)));
    gelu = register_module("gelu", torch::nn::GELU());
    down_sample = register_module("down_sample", torch::nn::Linear(torch::nn::LinearOptions(4 * n_embed, n_embed).bias(bias)));
    dropout_layer = register_module("dropout", torch::nn::Dropout(dropout));
}

torch::Tensor MLP::forward(const torch::Tensor& x) {
    auto output = this->up_sample->forward(x);
    output = this->gelu->forward(output);
    return this->dropout_layer->forward(this->down_sample->forward(output));
}

// Block Implementation
Block::Block(int64_t n_embed, int64_t n_head, float dropout, bool bias) {
    this->layer_norm_1 = register_module("layer_norm_1", std::make_shared<LayerNorm>(n_embed, bias));
    this->attention = register_module("attention", std::make_shared<SelfAttention>(n_embed, n_head, dropout, bias));
    this->layer_norm_2 = register_module("layer_norm_2", std::make_shared<LayerNorm>(n_embed, bias));
    this->mlp = register_module("mlp", std::make_shared<MLP>(n_embed, dropout, bias));
}

torch::Tensor Block::forward(const torch::Tensor& x) {
    auto out = x + this->attention->forward(this->layer_norm_1->forward(x));
    return out + this->mlp->forward(this->layer_norm_2->forward(out));
}


ChessModel::ChessModel(int64_t n_layer, int64_t n_head, int64_t n_embed, float dropout, bool bias) {
    this->n_layer = n_layer;
    this->n_head = n_head;
    this->n_embed = n_embed;
    this->dropout = dropout;
    this->bias = bias;

    this->piece_embed = register_module("piece_embed", torch::nn::Embedding(torch::nn::EmbeddingOptions(16, n_embed)));
    this->position_embed = register_module("position_embed", torch::nn::Embedding(torch::nn::EmbeddingOptions(64, n_embed)));
    this->castling_embed = register_module("castling_embed", torch::nn::Embedding(torch::nn::EmbeddingOptions(4, n_embed)));
    this->enpassant_embed = register_module("enpassant_embed", torch::nn::Embedding(torch::nn::EmbeddingOptions(1, n_embed)));
    this->repetition_embed = register_module("repetition_embed", torch::nn::Embedding(torch::nn::EmbeddingOptions(3, n_embed)));
    this->rule50_embed = register_module("rule50_embed", torch::nn::Embedding(torch::nn::EmbeddingOptions(50, n_embed)));

    // Set the embedding for indices 6, 7, 14, 15 to zero (not valid pieces)
    auto weights = this->piece_embed->weight;
    this->piece_embed->weight.data().index_put_({torch::tensor({6, 7, 14, 15})}, torch::zeros({n_embed}));
    this->piece_embed->weight.data().index({torch::tensor({6, 7, 14, 15})}).requires_grad_(false);


    for (int64_t i = 0; i < n_layer; ++i) {
        blocks.push_back(register_module("block_" + std::to_string(i), std::make_shared<Block>(n_embed, n_head, dropout, bias)));
    }

    this->layer_norm = register_module("layer_norm", std::make_shared<LayerNorm>(n_embed, bias));
    this->policy = register_module("policy", torch::nn::Linear(torch::nn::LinearOptions(n_embed, 1882).bias(bias)));
    this->evaluation = register_module("evaluation", torch::nn::Linear(torch::nn::LinearOptions(n_embed, 1).bias(bias)));
}


std::vector<torch::Tensor> ChessModel::forward(const torch::Tensor& x) {
    auto input = x;
    for (const auto& block : blocks) {
        input = block->forward(input);
    }
    input = this->layer_norm->forward(input);
    input = torch::mean(input, 1, false);

    torch::Tensor move_logits = this->policy->forward(input);
    torch::Tensor evaluation = this->evaluation->forward(input);

    std::vector<torch::Tensor> result;
    result.push_back(evaluation);
    result.push_back(move_logits);
    return result;
}


int relative_square(int sq, Color Us) {
    if (Us == WHITE) {
        return create_square(file_of(Square(sq)), relative_rank<WHITE>(rank_of(Square(sq))));
    } else {
        return create_square(file_of(Square(sq)), relative_rank<BLACK>(rank_of(Square(sq))));
    }
}

torch::Tensor ChessModel::board_to_tensor(const Board& board, const torch::Device device) {
    // Initialize a tensor to store the resulting embedding for each square
    auto tensor = torch::zeros({64, this->n_embed}, torch::kFloat32);

    Color Us = board.turn();

    for (int index = 0; index < 64; ++index) {

        int square = relative_square(index, Us);

        int piece_idx = board.piece_at(Square(square));
        if (piece_idx != NO_PIECE) {
            piece_idx ^= (Us << 3);
        }
        auto piece_emb =    this->piece_embed->forward(torch::tensor(piece_idx, torch::kLong));
        auto position_emb = this->position_embed->forward(torch::tensor(index, torch::kLong));
        auto r50_emb =      this->rule50_embed->forward(torch::tensor(std::clamp(board.get_rule_50(), 0, 49), torch::kLong));
        auto repe_emb =     this->repetition_embed->forward(torch::tensor(std::clamp(board.get_repetition()-1, 0, 2), torch::kLong));

        // Add the embeddings (piece + position) and store them in the tensor
        tensor.index_put_({index}, piece_idx);// + position_emb + r50_emb + repe_emb);

        // If the current piece is a king, handle castling rights
        if (type_of(Piece(piece_idx)) == KING) {

            if (color_of(Piece(piece_idx)) == Us) {
                if (board.can_cstle_king(Us)) {
                    auto emb = this->castling_embed->forward(torch::tensor(0, torch::kLong));
                    tensor.index_put_({square}, tensor.index({square}) + emb);
                }
                if (board.can_cstle_queen(Us)) {
                    auto emb = this->castling_embed->forward(torch::tensor(1, torch::kLong));
                    tensor.index_put_({square}, tensor.index({square}) + emb);
                }
            }
            else {    
                if (board.can_cstle_king(~Us)) {
                    auto emb = this->castling_embed->forward(torch::tensor(2, torch::kLong));
                    tensor.index_put_({square}, tensor.index({square}) + emb);
                }
                if (board.can_cstle_queen(~Us)) {
                    auto emb = this->castling_embed->forward(torch::tensor(3, torch::kLong));
                    tensor.index_put_({square}, tensor.index({square}) + emb);
                }
            }
        }
    }

    // Add en passant embedding if applicable
    int enpassant_sq = int(board.enpassant_square());
    if (enpassant_sq != NO_SQUARE) {
        enpassant_sq = relative_square(enpassant_sq, Us);
        auto emb = this->enpassant_embed->forward(torch::tensor(0, torch::kLong));
        tensor.index_put_({enpassant_sq}, tensor.index({enpassant_sq}) + emb);
    }

    return tensor;
}

int64_t ChessModel::get_num_params() const {
    size_t parameter_count = 0;

    // Iterate over all parameters of the embedding layer
    for (const auto& param : this->parameters()) {
        parameter_count += param.numel(); // Add the number of elements in each parameter tensor
    }

    return parameter_count;
}



#endif
