#include "monte_carlo.h"
#include "timer.h"
#include "model.h"
#include <cmath>
#include <algorithm>
#include <random>
#include <numeric>


template<typename T>
inline int random_index(const std::vector<T>& weights) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    float total = std::accumulate(weights.begin(), weights.end(), 0.0f);
    float random_value = dis(gen);

    float cumulative_sum = 0.0f;
    for (size_t i = 0; i < weights.size(); ++i) {
        cumulative_sum += weights[i] / total;
        if (random_value < cumulative_sum) {
            return i;
        }
    }
    return weights.size() - 1;
}

template<typename T>
inline int max_index(const std::vector<T>& vec) {
    if (vec.size() == 0) {
        throw std::invalid_argument("Input vector to max_index is empty.");
    }

    auto max_it = std::max_element(vec.begin(), vec.end());
    return std::distance(vec.begin(), max_it);
}

void calculateZScores(std::vector<float>& data) {
    // Step 1: Calculate the mean
    float sum = std::accumulate(data.begin(), data.end(), 0.0f);
    float mean = sum / data.size();

    // Step 2: Calculate the standard deviation
    float sq_sum = std::accumulate(data.begin(), data.end(), 0.0f, 
                                   [mean](float acc, float x) { return acc + (x - mean) * (x - mean); });
    float std_dev = std::sqrt(sq_sum / data.size());

    // Step 3: Transform each element into its Z-score
    for (float& x : data) {
        x = (x - mean) / std_dev;
    }
}



bool is_white_king_dead(Board& board) { return board.piece_bitboard(WHITE_KING) == 0; }
bool is_black_king_dead(Board& board) { return board.piece_bitboard(BLACK_KING) == 0; }
bool is_game_draw(Board& board) { return board.is_insufficient() || board.is_repetition() || board.is_rule_50(); }





MonteCarlo::MonteCarlo(Model& m, MonteCarloConfig config) : model(m), nodes_map{} {
    this->is_black_win = is_white_king_dead;
    this->is_white_win = is_black_king_dead;
    this->is_draw = is_game_draw;
    this->exploration_scale = config.exploration_scale;
    this->exploration_decay = config.exploration_decay;
    this->max_nodes = config.max_nodes;
    this->max_depth = config.max_depth;
}


MonteCarlo::MonteCarlo(Model& m) : MonteCarlo(m, MonteCarloConfig()) {}

inline Node& MonteCarlo::get_node(Board& board) {

    uint64_t hash = board.get_hash();

    if (this->nodes_map.find(hash) != this->nodes_map.end()) {
        return this->nodes_map[hash];
    }
    
    this->nodes_map[hash] = Node(); 
    return this->nodes_map[hash];
}


inline void MonteCarlo::roll_out(Board& board, Node& node) {

    node.legal_moves = board.get_legal_moves();
    node.move_weights = std::vector<float>(node.legal_moves.size(), 1);

    if (this->is_white_win(board)) {
        node.evaluation = 1;
        node.game_ended = true;
    } else if (this->is_black_win(board)) {
        node.evaluation = -1;
        node.game_ended = true;
    } else if (this->is_draw(board)) {
        node.evaluation = 0;
        node.game_ended = true;
    } else {
        
        node.evaluation = this->model(board, node.legal_moves, node.move_weights);
        std::transform(
            node.move_weights.begin(),
            node.move_weights.end(),
            node.move_weights.begin(),
            static_cast<float(*)(float)>(std::exp) 
        );
        //model returns logits, which can be negative so we can take e^x for positive values
        //we are essentially trying to compute softmax later on
    }
}



inline float MonteCarlo::node_weight(Node& node, int N, bool white_turn, int depth) { //uses modified UCB1
    if (node.visits == 0) {
        return INFINITY;
    }

    float exploitation = node.total/float(node.visits);
    float exploration = this->exploration_scale 
                        * std::exp(-depth * this->exploration_decay) 
                        * std::sqrt(std::log(N*2)/float(node.visits)); 
                        //for higher depth, we want to do less exploring and more exploiting
    
    if (white_turn) {
        return -exploitation + exploration;
    }
    return exploitation + exploration;
}



inline float MonteCarlo::visit(Board& board, int depth) {

    if (depth >= this->max_depth) {
        return 0;
    }
        

    Node& node = this->get_node(board);

    if (node.visits == 0) { //leaf node
        this->roll_out(board, node);
        node.visits++;
        node.total += node.evaluation;
        return node.evaluation;
    }

    
    if (node.game_ended | node.legal_moves.size() == 0) {
        node.visits++;
        node.total += node.evaluation;
        return node.evaluation;
    }

    std::vector<Move> unexplored_moves;
    std::vector<float> unexplored_weights;

    std::vector<Move> explored_moves;
    std::vector<float> explored_weights;

    //need to store all the variables before hand, since get_node can delete node pointers
    std::vector<Move> legal_moves = node.legal_moves;
    std::vector<float> move_weights = node.move_weights;
    uint32_t total_visits = node.visits;

    for (int i = 0; i < legal_moves.size(); i++) {
        Move move = legal_moves[i];
        board.play(move);
        Node& n = this->get_node(board);
        if (n.visits <= 0) {
            unexplored_moves.push_back(move);
            unexplored_weights.push_back(move_weights[i]);
        } else {
            float weight = this->node_weight(n, total_visits, board.turn()==WHITE, depth);
            explored_moves.push_back(move);
            explored_weights.push_back(weight);
        }

        board.undo(move);
    }

    Move best_move;

    if (unexplored_moves.size() > 0) {
        best_move = unexplored_moves[random_index<float>(unexplored_weights)];
    } else {
        best_move = explored_moves[max_index<float>(explored_weights)];
    }


    board.play(best_move);
    float eval = this->visit(board, depth+1); //back propagation
    board.undo(best_move);
    
    node = this->get_node(board);
    node.total += eval;
    node.visits++;

    return eval;
}


Move MonteCarlo::search(Board& board, int search_time_ms) {
    
    if (board.get_legal_moves().size() == 0) {
        throw std::invalid_argument("MonteCarlo.search() can not be called for positions with no legal moves");
    }
    
    this->iterations_searched = 0;

    if (this->is_draw(board) || this->is_black_win(board) || this->is_white_win(board)) {
        return Move();
    }


    Timer timer(search_time_ms);

    this->nodes_map = std::unordered_map<uint64_t, Node>();

    while (timer.time_remaining() > 0) {
        for (int i = 0; i < 100; i++) {
            this->visit(board, 0);
            this->iterations_searched++;
            if (this->iterations_searched >= this->max_nodes) break;
        }
        if (this->iterations_searched >= this->max_nodes) break;
    }
    
    std::vector<Move> legal_moves = board.get_legal_moves();
    std::vector<float> visits;

    for (Move m : legal_moves) {
        board.play(m);
        Node& child_node = this->get_node(board);
        board.undo(m);
        visits.push_back(child_node.visits);
    }

    //calculateZScores(visits);
    //
    //for (int i = 0; i < visits.size(); i++) {
    //    visits[i] = std::exp(std::min(visits[i] * 5.0f, 2.0f));
    //}

    return legal_moves[max_index<float>(visits)];
}  



int MonteCarlo::get_iterations_searched() {
    return this->iterations_searched;
}