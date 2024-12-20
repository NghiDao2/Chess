from board_wrapper import Move, Board
import math

class Node():

    def __init__(self):
        self.total = 0
        self.visits = 0
        self.legal_moves = None
        self.logits = None
        self.evaluation = None
        self.game_ended = False


class MonteCarlo():

    def __init__(self, model, iterations=50000, max_depth=250, max_nodes=500000):
        self.nodes = {}
        self.model = model
        self.nodes_visited = 0
        self.iterations = iterations
        self.max_depth=250

        self.node_pointer = 0
        self.max_nodes = max_nodes
        self.hash_list = [0]*max_nodes

    def search(self, board: Board):

        self.nodes_visited = 0

        for i in range(self.iterations):
            self.visit(board)

        # Print total iterations per second at the end

        node = self.get_node(board)

        if board.get_winner() != 0 or len(node.legal_moves) == 0:
            return None

        best = float("-inf")
        move = None

        for m in board.get_legal_moves():

            board.play(m)
            child_node = self.get_node(board)
            board.undo(m)

            #print(m, child_node.visits)

            score = child_node.visits
            if score > best:
                move = m
                best = score
        
        return move

    def get_node(self, board):
        h = board.get_hash()
        if not (h in self.nodes):
            
            if self.hash_list[self.node_pointer] in self.nodes:
                self.nodes.pop(self.hash_list[self.node_pointer])

            self.nodes[h] = Node()
            self.hash_list[self.node_pointer] = h
            self.node_pointer = (self.node_pointer + 1) % self.max_nodes

        return self.nodes[h]

    def roll_out(self, node: Node, board: Board):
        if node.legal_moves == None:
            node.legal_moves = board.get_legal_moves()

        node.game_ended = (
            board.get_winner() != 0
            or len(node.legal_moves) == 0
            or board.is_repetition()
            or board.is_insufficient())

        if node.game_ended:
            winner = board.get_winner()

            if winner != 0:
                if winner == -1:
                    node.evaluation = -1
                else:
                    node.evaluation = 1
            else:
                node.evaluation = 0
        else:
            if node.logits == None or node.evaluation == None:
                evaluation, logits = self.model(board, node.legal_moves)
                node.logits = logits
                node.evaluation = evaluation

        

    
    def node_weight(self, node, N, white_turn, depth): #uses modified UCB1
        if node.visits == 0:
            return float("inf")

        weight = node.total/node.visits
        adj =  math.exp(-depth/5) * math.sqrt(math.log(N*2)/node.visits)
        if white_turn:
            return -weight + adj
        return weight + adj


    def visit(self, board: Board, depth=1):
        
        self.nodes_visited += 1

        if depth >= self.max_depth:
            return 0 #if game carries on for too long, it is probably a tie

        node = self.get_node(board)

        if node.visits == 0: # leaf node
            self.roll_out(node, board)
            node.visits += 1
            node.total += node.evaluation
            return node.evaluation
        
        if node.game_ended:
            node.visits += 1
            node.total += node.evaluation
            return node.evaluation
        
        best_weight = float("-inf")
        move = None
        
        i = 0

        legals = node.legal_moves
        for m in legals:
            
            board.play(m)
            weight = self.node_weight(self.get_node(board), node.visits, board.is_white_turn(), depth) 
            board.undo(m)

            if weight > best_weight:
                move = m
                best_weight = weight
            i += 1

        board.play(move)
        evaluation = self.visit(board, depth+1) #back propagation
        board.undo(move)

        node.total += evaluation
        node.visits += 1

        return evaluation
    



