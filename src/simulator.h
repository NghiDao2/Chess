#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "monte_carlo.h"
#include "board.h"
#include "model.h"
#include "timer.h"
#include <string>
#include <vector>



class SimulatorConfig {
public:
    SimulatorConfig(MonteCarlo& white, MonteCarlo& black)
        : white_player(white), black_player(black) {}

    MonteCarlo& get_white_player() { return white_player; }
    MonteCarlo& get_black_player() { return black_player; }

    uint32_t move_time = 2000; 
    uint32_t move_limit = 400;

private:
    MonteCarlo& white_player;
    MonteCarlo& black_player;
};


class Simulator {
public:

    Simulator(SimulatorConfig config);
    
    void run(bool log = false);
    void save(const std::string& path,
            const std::string& filename,
            const std::string& white_name = "Bot", 
            const std::string& black_name = "Bot");

    int64_t get_time_elapsed();
    uint64_t get_total_iterations();
    vector<Move> get_move_sequence();

    bool is_white_win();
    bool is_black_win();
    bool is_draw();
    bool game_ended = false;

private:
    MonteCarlo& white_player;
    MonteCarlo& black_player;
    uint32_t move_time;
    uint32_t move_limit;
    Board board; //self.board = Board(starting_fen) 
    uint64_t total_iterations;
    int64_t time_elapsed;
    int winner;
    Timer timer;

    vector<Move> move_sequence;
};

#endif