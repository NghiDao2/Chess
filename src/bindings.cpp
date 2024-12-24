#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "model.h"
#include "board.h"
#include "timer.h"
#include "monte_carlo.h"
#include "simulator.h"
#include "simulator_batch.h"


void thread_function(TorchModel& model, int thread_id) {
    // Get legal moves
    Board board;
    std::vector<Move> legal_moves = board.get_legal_moves();
    std::vector<float> logits(legal_moves.size(), 0);

    for (int j = 0; j < 100; ++j) {
        // Perform the evaluation
        float eval = model(board, legal_moves, logits);
    }
}

namespace py = pybind11;

void initialize_all() {
    initialise_all_databases();           // Ensure your function is declared and accessible
    zobrist::initialise_zobrist_keys();  // Call Zobrist initialization





    try {
        // Test LayerNorm
        std::cout << "Testing LayerNorm..." << std::endl;
        LayerNorm layer_norm(64);
        torch::Tensor ln_input = torch::randn({8, 64}); // Batch size 8, feature size 64
        torch::Tensor ln_output = layer_norm.forward(ln_input);
        std::cout << "LayerNorm output: " << ln_output.sizes() << std::endl;

        // Test SelfAttention
        std::cout << "\nTesting SelfAttention..." << std::endl;
        SelfAttention self_attention(64, 8); // 64-dimensional embedding, 8 attention heads
        torch::Tensor sa_input = torch::randn({8, 16, 64}); // Batch size 8, sequence length 16, embedding size 64
        torch::Tensor sa_output = self_attention.forward(sa_input);
        std::cout << "SelfAttention output: " << sa_output.sizes() << std::endl;

        // Test MLP
        std::cout << "\nTesting MLP..." << std::endl;
        MLP mlp(64);
        torch::Tensor mlp_input = torch::randn({8, 64}); // Batch size 8, feature size 64
        torch::Tensor mlp_output = mlp.forward(mlp_input);
        std::cout << "MLP output: " << mlp_output.sizes() << std::endl;

        // Test Block
        std::cout << "\nTesting Block..." << std::endl;
        Block block(64, 8); // 64-dimensional embedding, 8 attention heads
        torch::Tensor block_input = torch::randn({8, 16, 64}); // Batch size 8, sequence length 16, embedding size 64
        torch::Tensor block_output = block.forward(block_input);
        std::cout << "Block output: " << block_output.sizes() << std::endl;

        // Test ChessModel
        std::cout << "\nTesting ChessModel..." << std::endl;
        ChessModel chess_model(4, 1, 1); 
        torch::Tensor chess_input = torch::randn({8, 8, 1}); 
        std::vector<torch::Tensor> chess_output = chess_model.forward(chess_input);
        std::cout << "ChessModel policy output: " << chess_output[0].sizes() << std::endl;
        std::cout << "ChessModel evaluation output: " << chess_output[1].sizes() << std::endl;

        // Print number of parameters in ChessModel
        int64_t num_params = chess_model.get_num_params();
        std::cout << "\nNumber of parameters in ChessModel: " << num_params << std::endl;



        Board board("rnbqkbnr/pppppppp/8/8/6PP/8/PPPPPP2/RNBQKBNR w KQkq g3 0 1");
        std::cout << "board 1" << std::endl;
        torch::Tensor board_tensor = chess_model.board_to_tensor(board);

        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                std::cout << board_tensor[y*8+x].item<float>() << " | ";
            }
            std::cout << std::endl;
        }


        Board board2("rnbqkbnr/pppppp2/8/6pp/8/8/PPPPPPPP/RNBQKBNR b KQkq e3 0 1");
        std::cout << "board 1" << std::endl;
        torch::Tensor board_tensor2 = chess_model.board_to_tensor(board2);

        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                std::cout << board_tensor2[y*8+x].item<float>() << " | ";
            }
            std::cout << std::endl;
        }




        /*
        
        std::cout << "Testing TorchModel..." << std::endl;

        ModelConfig conf;
        conf.n_embed = 64;
        conf.n_head = 8;
        conf.n_layer = 4;
        TorchModel model(conf);

        
        Timer timer(1000000);

        model.eval_mode();

        const int num_threads = 100;
        std::vector<std::thread> threads;

        // Launch 10 threads
        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back(thread_function, std::ref(model), i); // Pass model by reference
        }

        // Join all threads to wait for completion
        for (std::thread& t : threads) {
            t.join();
        }

        std::cout << "Elapsed time for 10,000 evaluations: " 
              << timer.time_elapsed() << " seconds" << std::endl;
        
        */

        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }





}


PYBIND11_MODULE(wrapper, m) {
    m.doc() = "Python bindings for Chess Engine";

    m.def("initialise_all_databases", &initialise_all_databases, "Initializes all databases.");
    m.def("initialise_zobrist_keys", &zobrist::initialise_zobrist_keys, "Initializes Zobrist keys.");
    m.def("initialize_all", &initialize_all, "Initializes all required databases and Zobrist keys.");


    // Bind the Piece enum
    py::enum_<Piece>(m, "Piece")
        .value("WHITE_PAWN", WHITE_PAWN)
        .value("WHITE_ROOK", WHITE_ROOK)
        .value("WHITE_KNIGHT", WHITE_KNIGHT)
        .value("WHITE_BISHOP", WHITE_BISHOP)
        .value("WHITE_QUEEN", WHITE_QUEEN)
        .value("WHITE_KING", WHITE_KING)
        .value("BLACK_PAWN", BLACK_PAWN)
        .value("BLACK_ROOK", BLACK_ROOK)
        .value("BLACK_KNIGHT", BLACK_KNIGHT)
        .value("BLACK_BISHOP", BLACK_BISHOP)
        .value("BLACK_QUEEN", BLACK_QUEEN)
        .value("BLACK_KING", BLACK_KING)
        .export_values();

    // Bind the Move class
    py::class_<Move>(m, "Move")
        .def(py::init<>())
        .def(py::init<uint16_t>())
        .def(py::init<Square, Square>())
        .def(py::init<Square, Square, MoveFlags>())
        .def(py::init<const std::string&>())
        .def("from_square", &Move::from)
        .def("to_square", &Move::to)
        .def("is_capture", &Move::is_capture)
        .def("is_promotion", &Move::is_promotion)
        .def("is_enpassant", &Move::is_enpassant)
        .def("to_int16", &Move::to_int16)
        .def("__str__", &Move::to_string)
        .def("__eq__", &Move::operator==)
        .def("__ne__", &Move::operator!=);

    // Bind the Board class
    py::class_<Board>(m, "Board")
        .def(py::init<>())
        .def(py::init<std::string>())
        .def("play", &Board::play, py::arg("move"))
        .def("undo", &Board::undo, py::arg("move"))
        .def("__str__", &Board::to_string)
        .def("get_legal_moves", &Board::get_legal_moves)
        .def("piece_bitboard", &Board::piece_bitboard)
        .def("is_white_turn", &Board::is_white_turn)
        .def("is_repetition", &Board::is_repetition)
        .def("is_insufficient", &Board::is_insufficient)
        .def("is_rule_50", &Board::is_rule_50);


    py::class_<Model, std::shared_ptr<Model>>(m, "Model");

    // Register DefaultEvaluation as a subclass of Model with std::shared_ptr as the holder type
    py::class_<DefaultEvaluation, Model, std::shared_ptr<DefaultEvaluation>>(m, "DefaultEvaluation")
        .def(py::init<>())  // Default constructor
        .def("__call__", [](DefaultEvaluation& eval, Board& board, std::vector<Move>& legal_moves) {
            std::vector<float> logits(legal_moves.size(), 1.0f);
            float eval_result = eval(board, legal_moves, logits);
            return py::make_tuple(eval_result, logits);
        }, py::arg("board"), py::arg("legal_moves"));



    py::class_<MonteCarloConfig>(m, "MonteCarloConfig")
        .def(py::init<>())  // Default constructor
        .def_readwrite("exploration_scale", &MonteCarloConfig::exploration_scale)
        .def_readwrite("exploration_decay", &MonteCarloConfig::exploration_decay)
        .def_readwrite("max_nodes", &MonteCarloConfig::max_nodes)
        .def_readwrite("max_depth", &MonteCarloConfig::max_depth);


    // Bind MonteCarlo class instantiated with DefaultEvaluation
    py::class_<MonteCarlo>(m, "MonteCarlo")
        .def(py::init<Model&>(), py::arg("model"))
        .def(py::init<Model&, MonteCarloConfig&>(), py::arg("model"), py::arg("config"))
        .def("search", &MonteCarlo::search, py::arg("board"), py::arg("search_time_ms"),
             "Perform a Monte Carlo search to determine the best move")
        .def("get_iterations_searched", &MonteCarlo::get_iterations_searched,
             "Get the total number of iterations searched");


    py::class_<SimulatorConfig>(m, "SimulatorConfig")
        .def(py::init<MonteCarlo&, MonteCarlo&>(), py::arg("white_player"), py::arg("black_player"))
        .def_property_readonly("white_player", &SimulatorConfig::get_white_player)
        .def_property_readonly("black_player", &SimulatorConfig::get_black_player)
        .def_readwrite("move_time", &SimulatorConfig::move_time)
        .def_readwrite("move_limit", &SimulatorConfig::move_limit);


    py::class_<Simulator>(m, "Simulator")
        .def(py::init<SimulatorConfig>(), py::arg("config"))
        .def("run", &Simulator::run, py::arg("log") = false)
        .def("save", &Simulator::save, py::arg("path"), py::arg("filename"), 
        py::arg("white_name") = "Bot", py::arg("black_name") = "Bot")
        
        .def("get_time_elapsed", &Simulator::get_time_elapsed)
        .def("get_total_iterations", &Simulator::get_total_iterations)
        .def("get_move_sequence", &Simulator::get_move_sequence)
        .def("is_white_win", &Simulator::is_white_win)
        .def("is_black_win", &Simulator::is_black_win)
        .def("is_draw", &Simulator::is_draw);


    py::class_<SimulatorBatch>(m, "SimulatorBatch")
        .def(py::init<int>(), py::arg("num_processes") = 4) 
        .def("total_remaining", &SimulatorBatch::total_remaining)  
        .def("add", &SimulatorBatch::add, py::arg("game")); 


#ifdef HAS_TORCH
    // Bind ModelConfig
    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("n_layer", &ModelConfig::n_layer)
        .def_readwrite("n_head", &ModelConfig::n_head)
        .def_readwrite("n_embed", &ModelConfig::n_embed)
        .def_readwrite("dropout", &ModelConfig::dropout)
        .def_readwrite("bias", &ModelConfig::bias);

    // Bind TorchModel
    py::class_<TorchModel, Model, std::shared_ptr<TorchModel>>(m, "TorchModel")
        .def(py::init<ModelConfig>(), py::arg("config"))
        .def("__call__", [](TorchModel& eval, Board& board, std::vector<Move>& legal_moves) {
            std::vector<float> logits(legal_moves.size(), 1.0f);
            float eval_result = eval(board, legal_moves, logits);
            return py::make_tuple(eval_result, logits);
        }, py::arg("board"), py::arg("legal_moves"));

#endif
}
