#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <vector>
#include <array>
#include <unordered_map>
#include <functional>
#include "board.h"



class MonteCarloConfig {
public:
    MonteCarloConfig(): 
        exploration_scale(3.5),
        exploration_decay(0.25),
        max_nodes(4194304),
        max_depth(256) {}

    float exploration_scale;
    float exploration_decay;
    int max_nodes;
    int max_depth;
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


template <typename Model>
class MonteCarlo {

public:
    MonteCarlo(Model& m);
    MonteCarlo(Model& m, MonteCarloConfig config);
    Move search(Board& board, int search_time_ms);
    int get_iterations_searched();
    
private:
    Model model;
    int iterations_searched;
    float exploration_scale; //how strongly the monte carlo chooses exploration over exploitation
    float exploration_decay; //for high depth search, the model should prioritize exploitation over exploration?
    int max_nodes;
    int max_depth;

    std::unordered_map<uint64_t, Node> nodes_map{};

    std::function<bool(Board&)> is_black_win;
    std::function<bool(Board&)> is_white_win;
    std::function<bool(Board&)> is_tie;

    inline void roll_out(Board& board, Node& node);
    inline Node& get_node(Board& board);
    inline Node& roll_out(Board& board);
    inline float node_weight(Node& node, int N, bool white_turn, int depth);
    float visit(Board& board, int depth);
};



#endif 