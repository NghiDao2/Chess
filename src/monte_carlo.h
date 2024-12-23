#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <vector>
#include <array>
#include <unordered_map>
#include <functional>
#include "board.h"
#include "model.h"

bool is_white_king_dead(Board& board);
bool is_black_king_dead(Board& board);
bool is_game_draw(Board& board);



class MonteCarloConfig {
public:
    float exploration_scale = 1.05;
    float exploration_decay = 0.45;
    int max_nodes = 4194304;
    int max_depth = 256;
};

class Node {

public:
    std::vector<Move> legal_moves;
    std::vector<float> move_weights;
    uint32_t visits;
    float total;
    float evaluation;
    bool game_ended;
private:

};


class MonteCarlo {

public:
    MonteCarlo(Model& m);
    MonteCarlo(Model& m, MonteCarloConfig config);
    Move search(Board& board, int search_time_ms);
    int get_iterations_searched();
    
private:
    Model& model;
    int iterations_searched;
    float exploration_scale; //how strongly the monte carlo chooses exploration over exploitation
    float exploration_decay; //for high depth search, the model should prioritize exploitation over exploration?
    int max_nodes;
    int max_depth;

    std::unordered_map<uint64_t, Node> nodes_map{};

    std::function<bool(Board&)> is_black_win;
    std::function<bool(Board&)> is_white_win;
    std::function<bool(Board&)> is_draw;

    inline void roll_out(Board& board, Node& node);
    inline Node& get_node(Board& board);
    inline Node& roll_out(Board& board);
    inline float node_weight(Node& node, int N, bool white_turn, int depth);
    float visit(Board& board, int depth);
};



#endif 