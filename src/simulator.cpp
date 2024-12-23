
#include "simulator.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>



Simulator::Simulator(SimulatorConfig config) : 
white_player(config.get_white_player()), 
black_player(config.get_black_player()),
move_time(config.move_time),
move_limit(config.move_limit),
board(DEFAULT_FEN),
timer(0xFFFFFFFFFFFF) {

}


bool Simulator::is_white_win() {return this->winner == 1 && this->game_ended;}
bool Simulator::is_black_win() {return this->winner == -1 && this->game_ended;}
bool Simulator::is_draw() {return this->winner == 0 && this->game_ended;}

void Simulator::run(bool log) {

    int64_t start_time = this->timer.time_elapsed();

    if (log) {
        std::cout << board.to_string() << std::endl;
    }

    while (true) {
        if (move_sequence.size() > move_limit) {
            break;  // Assume a draw if move limit is reached
        }
        if (is_game_draw(board)) {
            break;
        }
        if (board.get_legal_moves().size() == 0) {
            break;
        }
        if (is_black_king_dead(board)) {
            winner = 1;
            break;
        }
        if (is_white_king_dead(board)) {
            winner = -1;
            break;
        }

        int iterations = 0;
        if (board.is_white_turn()) {
            Move m = white_player.search(board, move_time);
            board.play(m);
            move_sequence.push_back(m);
            iterations = white_player.get_iterations_searched();
        } else {
            Move m = black_player.search(board, move_time);
            board.play(m);
            move_sequence.push_back(m);
            iterations = black_player.get_iterations_searched();
        }

        total_iterations += iterations;

        if (log) {
            std::cout << board.to_string() << std::endl;
            std::cout << "Searched " << iterations << " iterations" << std::endl;
            std::cout << "Move: " << move_sequence.back() << std::endl;
        }
    }

    int64_t end_time = this->timer.time_elapsed();
    this->time_elapsed = end_time - start_time;
    this->game_ended = true;
}


int64_t Simulator::get_time_elapsed() {
    return this->time_elapsed;
}

uint64_t Simulator::get_total_iterations() {
    return this->total_iterations;
}

std::vector<Move> Simulator::get_move_sequence() {
    return this->move_sequence;
}


void Simulator::save(  const std::string& path,
            const std::string& filename,
            const std::string& white_name, 
            const std::string& black_name) {

    std::filesystem::create_directories(path);
    std::string file_path = path + "/" + filename + ".txt";

    std::ofstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file for writing: " + file_path);
    }

    file << "[White \"" << white_name << "\"]\n";
    file << "[Black \"" << black_name << "\"]\n";
    file << "[Time per move " << move_time << "]\n";
    file << "[Time elapsed " << this->time_elapsed << "]\n";
    file << "[Total iterations " << this->total_iterations << "]\n";

    if (winner == 1) {
        file << "[Result 1-0]\n";
    } else if (winner == -1) {
        file << "[Result 0-1]\n";
    } else {
        file << "[Result 0-0]\n";
    }

    file << "\n";
    for (size_t i = 0; i < this->move_sequence.size(); ++i) {
        if (i % 2 == 0) {
            file << i / 2 + 1 << ". " << this->move_sequence[i] << " ";
        } else {
            file << this->move_sequence[i] << " ";
        }
    }

    if (winner == 1) {
        file << "1-0\n";
    } else if (winner == -1) {
        file << "0-1\n";
    } else {
        file << "0-0\n";
    }
}