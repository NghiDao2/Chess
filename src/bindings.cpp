#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "model.h"
#include "board.h"
#include "monte_carlo.h"

namespace py = pybind11;

void initialize_all() {
    initialise_all_databases();           // Ensure your function is declared and accessible
    zobrist::initialise_zobrist_keys();  // Call Zobrist initialization
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

    // Bind DefaultEvaluation
    py::class_<DefaultEvaluation>(m, "DefaultEvaluation")
        .def(py::init<>())
        .def("__call__", [](DefaultEvaluation& eval, Board& board, std::vector<Move>& legal_moves, std::vector<float>& logits) {
            return eval(board, legal_moves, logits);
        }, "Evaluate the position", py::arg("board"), py::arg("legal_moves"), py::arg("logits"));


    py::class_<MonteCarloConfig>(m, "MonteCarloConfig")
        .def(py::init<>())  // Default constructor
        .def_readwrite("exploration_scale", &MonteCarloConfig::exploration_scale)
        .def_readwrite("exploration_decay", &MonteCarloConfig::exploration_decay)
        .def_readwrite("max_nodes", &MonteCarloConfig::max_nodes)
        .def_readwrite("max_depth", &MonteCarloConfig::max_depth);

    // Bind MonteCarlo class instantiated with DefaultEvaluation
    py::class_<MonteCarlo<DefaultEvaluation>>(m, "MonteCarlo")
        .def(py::init<DefaultEvaluation&>(), py::arg("model"))
        .def(py::init<DefaultEvaluation&, MonteCarloConfig&>(), py::arg("model"), py::arg("config"))
        .def("search", &MonteCarlo<DefaultEvaluation>::search, py::arg("board"), py::arg("search_time_ms"),
             "Perform a Monte Carlo search to determine the best move")
        .def("get_iterations_searched", &MonteCarlo<DefaultEvaluation>::get_iterations_searched,
             "Get the total number of iterations searched");

#ifdef HAS_TORCH
    // Bind ModelConfig
    py::class_<ModelConfig>(m, "ModelConfig")
        .def(py::init<int, int, int, int>(),
             py::arg("n_layers"), py::arg("embed_dim"), py::arg("hidden_dim"), py::arg("output_dim"))
        .def_readwrite("n_layers", &ModelConfig::n_layers)
        .def_readwrite("embed_dim", &ModelConfig::embed_dim)
        .def_readwrite("hidden_dim", &ModelConfig::hidden_dim)
        .def_readwrite("output_dim", &ModelConfig::output_dim);

    // Bind TorchModel
    py::class_<TorchModel>(m, "TorchModel")
        .def(py::init<ModelConfig>(), py::arg("config"))
         .def("__call__", [](TorchModel& eval, Board& board, std::vector<Move>& legal_moves, std::vector<float>& logits) {
            return eval(board, legal_moves, logits);
        }, "Evaluate the position", py::arg("board"), py::arg("legal_moves"), py::arg("logits"));


     // Bind MonteCarlo class instantiated with TorchModel
    py::class_<MonteCarlo<TorchModel>>(m, "MonteCarloTorch")
        .def(py::init<TorchModel&>(), py::arg("model"))
        .def(py::init<TorchModel&, MonteCarloConfig&>(), py::arg("model"), py::arg("config"))
        .def("search", &MonteCarlo<TorchModel>::search, py::arg("board"), py::arg("search_time_ms"))
        .def("get_iterations_searched", &MonteCarlo<TorchModel>::get_iterations_searched);
#endif
}
