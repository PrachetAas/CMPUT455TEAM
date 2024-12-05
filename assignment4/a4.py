# CMPUT 455 Assignment 4 starter code
# Implement the specified commands to complete the assignment
# Full assignment specification here: https://webdocs.cs.ualberta.ca/~mmueller/courses/cmput455/assignments/a4.html

# CMPUT 455 Assignment 4 starter code
# Implement the specified commands to complete the assignment

import sys
import random
import signal
import time
import math
import numpy as np
import traceback  # Added to print exceptions


# Custom time out exception
class TimeoutException(Exception):
    pass


# Function that is called when we reach the time limit
def handle_alarm(signum, frame):
    raise TimeoutException


class CommandInterface:

    def __init__(self):
        # Define the string to function command mapping
        self.command_dict = {
            "help": self.help,
            "game": self.game,
            "show": self.show,
            "play": self.play,
            "legal": self.legal,
            "genmove": self.genmove,
            "winner": self.winner,
            "timelimit": self.timelimit,
            # Add missing commands
            "set_opponent": self.set_opponent,
            "set_student_as_player": self.set_student_as_player,
            "?play_game": self.play_game
        }
        self.board = [[None]]
        self.player = 1
        self.max_genmove_time = 1
        # Define pattern_dict with dynamic patterns
        self.pattern_dict = {
            # Illegal patterns (heavily penalized)
            ('PPP',): -1000,  # Player triple (violates rule)
            ('OOO',): -1000,  # Opponent triple (violates rule)

            # Opponent's potential triples (block them)
            ('OO.',): 100,
            ('.OO',): 100,
            ('O.O',): 100,

            # Player's potential triples (encourage them)
            ('PP.',): 50,
            ('.PP',): 50,
            ('P.P',): 50,

            # Neutral patterns
            ('...',): 10,

            # Blocking patterns (neutralize threats)
            ('P.O',): 20,
            ('O.P',): 20,
            ('.P.',): 30,
            ('.O.',): 30,
        }

        self.transposition_table = {}
        self.zobrist_table = {}
        self.hash_value = 0
        self.eval_cache = {}
        self.start_time = 0
        signal.signal(signal.SIGALRM, handle_alarm)

    # ====================================================================================================================
    # VVVVVVVVVV Start of predefined functions. You may modify, but make sure not to break the functionality. VVVVVVVVVV
    # ====================================================================================================================

    # Convert a raw string to a command and a list of arguments
    def process_command(self, str):
        str = str.lower().strip()
        command = str.split(" ")[0]
        args = [x for x in str.split(" ")[1:] if len(x) > 0]
        if command not in self.command_dict:
            print("? Unknown command.\nType 'help' to list known commands.", file=sys.stderr)
            print("= -1\n")
            return False
        try:
            return self.command_dict[command](args)
        except Exception as e:
            print(f"Command '{str}' failed with exception:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("= -1\n")
            return False

    def copy(self):
        new_interface = CommandInterface()
        # Deep copy the board
        new_interface.board = [row.copy() for row in self.board]
        # Copy the player
        new_interface.player = self.player
        # Copy other necessary attributes
        new_interface.max_genmove_time = self.max_genmove_time
        new_interface.zobrist_table = self.zobrist_table
        new_interface.hash_value = self.hash_value
        new_interface.eval_cache = self.eval_cache.copy()
        new_interface.pattern_dict = self.pattern_dict
        new_interface.history = self.history.copy()
        return new_interface

    # Will continuously receive and execute commands
    def main_loop(self):
        while True:
            try:
                str = input()
            except EOFError:
                break
            if str.strip() == "":
                continue  # Skip empty lines
            if str.split(" ")[0] == "exit":
                print("= 1\n")
                return True
            if self.process_command(str):
                print("= 1\n")

    # Will make sure there are enough arguments, and that they are valid numbers
    def arg_check(self, args, template):
        converted_args = []
        if len(args) < len(template.split(" ")):
            print("Not enough arguments.\nExpected arguments:", template, file=sys.stderr)
            print("Received arguments: ", end="", file=sys.stderr)
            for a in args:
                print(a, end=" ", file=sys.stderr)
            print(file=sys.stderr)
            return False
        for i, arg in enumerate(args):
            try:
                converted_args.append(int(arg))
            except ValueError:
                print("Argument '" + arg + "' cannot be interpreted as a number.\nExpected arguments:", template,
                      file=sys.stderr)
                return False
        args = converted_args
        return True

    # List available commands
    def help(self, args):
        for command in self.command_dict:
            if command != "help":
                print(command)
        print("exit")
        return True

    def game(self, args):
        if not self.arg_check(args, "n m"):
            return False
        n, m = [int(x) for x in args]
        if n < 0 or m < 0:
            print("Invalid board size:", n, m, file=sys.stderr)
            return False

        self.board = []
        for i in range(m):
            self.board.append([None] * n)
        self.player = 1

        self.zobrist_table = {}
        self.hash_value = 0
        self.transposition_table = {}
        self.history = []

        # Initialize new Zobrist table
        self.initialize_zobrist_hash()
        return True

    def show(self, args):
        for row in self.board:
            for x in row:
                if x is None:
                    print(".", end="")
                else:
                    print(x, end="")
            print()
        return True

    def is_legal(self, x, y, num):
        if self.board[y][x] is not None:
            return False, "occupied"

        consecutive = 0
        count = 0
        self.board[y][x] = num
        for row in range(len(self.board)):
            if self.board[row][x] == num:
                count += 1
                consecutive += 1
                if consecutive >= 3:
                    self.board[y][x] = None
                    return False, "three in a row"
            else:
                consecutive = 0
        too_many = count > len(self.board) // 2 + len(self.board) % 2

        consecutive = 0
        count = 0
        for col in range(len(self.board[0])):
            if self.board[y][col] == num:
                count += 1
                consecutive += 1
                if consecutive >= 3:
                    self.board[y][x] = None
                    return False, "three in a row"
            else:
                consecutive = 0
        if too_many or count > len(self.board[0]) // 2 + len(self.board[0]) % 2:
            self.board[y][x] = None
            return False, "too many " + str(num)

        self.board[y][x] = None
        return True, ""

    def valid_move(self, x, y, num):
        if x >= 0 and x < len(self.board[0]) and \
                y >= 0 and y < len(self.board) and \
                (num == 0 or num == 1):
            legal, _ = self.is_legal(x, y, num)
            return legal

    def play(self, args):
        err = ""
        if len(args) != 3:
            print("= illegal move: " + " ".join(args) + " wrong number of arguments\n")
            return False
        try:
            x = int(args[0])
            y = int(args[1])
        except ValueError:
            print("= illegal move: " + " ".join(args) + " wrong coordinate\n")
            return False
        if x < 0 or x >= len(self.board[0]) or y < 0 or y >= len(self.board):
            print("= illegal move: " + " ".join(args) + " wrong coordinate\n")
            return False
        if args[2] != '0' and args[2] != '1':
            print("= illegal move: " + " ".join(args) + " wrong number\n")
            return False
        num = int(args[2])
        legal, reason = self.is_legal(x, y, num)
        if not legal:
            print("= illegal move: " + " ".join(args) + " " + reason + "\n")
            return False
        self.board[y][x] = num
        self.update_zobrist_hash(x, y, num)
        self.history.append((y, x, num))
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1
        print(f"Board after play {args}:\n{self.board_to_string()}", file=sys.stderr)
        return True

    def legal(self, args):
        if not self.arg_check(args, "x y number"):
            return False
        x, y, num = [int(x) for x in args]
        if self.valid_move(x, y, num):
            print("yes")
        else:
            print("no")
        return True

    def get_legal_moves(self):
        moves = []
        for y in range(len(self.board)):
            for x in range(len(self.board[0])):
                for num in range(2):
                    legal, _ = self.is_legal(x, y, num)
                    if legal:
                        moves.append([str(x), str(y), str(num)])
        return moves

    def winner(self, args):
        if len(self.get_legal_moves()) == 0:
            if self.player == 1:
                print(2)
            else:
                print(1)
        else:
            print("unfinished")
        return True

    def timelimit(self, args):
        self.max_genmove_time = int(args[0])
        return True

    # ===============================================================================================
    # ɅɅɅɅɅɅɅɅɅɅ End of predefined functions. ɅɅɅɅɅɅɅɅɅɅ
    # ===============================================================================================

    # ===============================================================================================
    # VVVVVVVVVV Start of Assignment 4 functions. Add/modify as needed. VVVVVVVV
    # ===============================================================================================

    # Set opponent command
    def set_opponent(self, args):
        if len(args) != 1:
            print("Error: set_opponent requires one argument (the opponent's script filename).", file=sys.stderr)
            return False
        self.opponent_script = args[0]
        return True

    # Set student as player command
    def set_student_as_player(self, args):
        if len(args) != 1 or args[0] not in ['1', '2']:
            print("Error: set_student_as_player requires one argument (1 or 2).", file=sys.stderr)
            return False
        self.student_player_num = int(args[0])
        return True

    # Play game command
    def play_game(self, args):
        # Check if opponent and student player number are set
        if not hasattr(self, 'opponent_script'):
            print("Error: Opponent script not set. Use set_opponent command.", file=sys.stderr)
            return False
        if not hasattr(self, 'student_player_num'):
            print("Error: Student player number not set. Use set_student_as_player command.", file=sys.stderr)
            return False

        import subprocess

        # Start the opponent process
        opponent_process = subprocess.Popen(
            ['python3', '-u', self.opponent_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line-buffered
        )

        current_player = 1
        game_over = False

        while not game_over:
            print(f"Current player: {current_player}", file=sys.stderr)
            if current_player == self.student_player_num:
                # Your player's turn
                move = self.genmove([])
                print(f"Your move: {move}", file=sys.stderr)
                if move == "resign":
                    opponent_process.stdin.write("resign\n")
                    opponent_process.stdin.flush()
                    opponent_response = self.read_opponent_response(opponent_process)
                    print(f"Opponent response after resign: {opponent_response}", file=sys.stderr)
                    print(3 - self.student_player_num)
                    game_over = True
                else:
                    # Apply your move
                    self.play(move.strip().split())
                    print(f"Applied your move: {move}", file=sys.stderr)
                    # Send your move to the opponent
                    opponent_process.stdin.write(f"play {move}\n")
                    opponent_process.stdin.flush()
                    opponent_response = self.read_opponent_response(opponent_process)
                    print(f"Opponent response after play: {opponent_response}", file=sys.stderr)
            else:
                # Opponent's turn
                opponent_process.stdin.write("genmove\n")
                opponent_process.stdin.flush()
                opponent_move = self.read_opponent_response(opponent_process)
                print(f"Opponent move: {opponent_move}", file=sys.stderr)
                if opponent_move == "resign":
                    print(self.student_player_num)
                    game_over = True
                else:
                    # Apply opponent's move
                    self.play(opponent_move.strip().split())
                    print(f"Applied opponent's move: {opponent_move}", file=sys.stderr)
                    # Send the move back to the opponent to keep game states in sync
                    opponent_process.stdin.write(f"play {opponent_move}\n")
                    opponent_process.stdin.flush()
                    opponent_response = self.read_opponent_response(opponent_process)
                    print(f"Opponent response after play: {opponent_response}", file=sys.stderr)

            # Check for game over
            if self.is_terminal():
                winner = 3 - current_player
                print(winner)
                game_over = True

            current_player = 3 - current_player  # Switch player

        opponent_process.terminate()
        return True

    def read_opponent_response(self, process):
        response = ''
        while True:
            line = process.stdout.readline()
            if not line:
                break  # End of output
            response += line
            if line.strip() == '= 1' or line.strip() == '= -1':
                break
        # Extract the move from the response
        lines = response.strip().split('\n')
        move = ''
        for line in lines:
            line = line.strip()
            if line and not line.startswith('='):
                move = line
                break
        print(f"Opponent response: {response}", file=sys.stderr)
        return move

    # MCTS Integration with Pattern-Based Enhancements
    def genmove(self, args):
        self.start_time = time.time()
        try:
            moves = self.get_legal_moves()
            if len(moves) == 0:
                print("resign")
                return "resign"

            best_move = None
            best_weight = float('-inf')

            # Evaluate each move and select the best one
            for move in moves:
                x, y, num = int(move[0]), int(move[1]), int(move[2])
                weight = self.calculate_move_weight(x, y, num)
                if weight > best_weight:
                    best_weight = weight
                    best_move = move

            if best_move:
                self.perform_move(best_move)
                move_str = " ".join(best_move)
                print(move_str)
                return move_str
            else:
                rand_move = moves[random.randint(0, len(moves) - 1)]
                self.perform_move(rand_move)
                move_str = " ".join(rand_move)
                print(move_str)
                return move_str

        except TimeoutException:
            print("resign")
            return "resign"
        except Exception as e:
            print(f"Exception in genmove: {e}", file=sys.stderr)
            print("resign")
            return "resign"


    # Additional Helper Functions (Zobrist, Board Handling)
    def initialize_zobrist_hash(self):
        row_range = len(self.board)
        col_range = len(self.board[0])
        for x in range(col_range):
            for y in range(row_range):
                self.zobrist_table[(x, y)] = [random.getrandbits(64) for _ in range(2)]

    def update_zobrist_hash(self, x, y, piece):
        self.hash_value ^= self.zobrist_table[(x, y)][piece]

    def perform_move(self, move):
        x, y, num = int(move[0]), int(move[1]), int(move[2])
        legal, reason = self.is_legal(x, y, num)
        if not legal:
            print(f"Attempted to perform illegal move: {move} Reason: {reason}", file=sys.stderr)
            return False
        self.board[y][x] = num
        self.update_zobrist_hash(x, y, num)
        self.history.append((y, x, num))
        self.player = 2 if self.player == 1 else 1
        print(f"Board after move {move}:\n{self.board_to_string()}", file=sys.stderr)
        return True

    def board_to_string(self):
        return '\n'.join([''.join(['.' if cell is None else str(cell) for cell in row]) for row in self.board])

    # Terminal Check Function
    def is_terminal(self):
        return len(self.get_legal_moves()) == 0

    # Get Result Function
    def get_result(self, player_num):
        if self.is_terminal():
            return -1 if self.player == player_num else 1
        return 0

    # Calculate Move Weight with Dynamic Patterns
    def calculate_move_weight(self, x, y, num):
        """Calculate the weight of a move at (x, y) for 'num' (0 or 1)."""
        weight = 0

        # Place the number temporarily
        self.board[y][x] = num

        player_num = num
        opponent_num = 1 - num

        # Function to map cell values to symbols
        def cell_to_symbol(cell):
            if cell is None:
                return '.'
            elif cell == player_num:
                return 'P'
            elif cell == opponent_num:
                return 'O'
            else:
                return '?'

    # Extract row and column patterns with symbols
        row = ''.join([cell_to_symbol(cell) for cell in self.board[y]])
        col = ''.join([cell_to_symbol(self.board[i][x]) for i in range(len(self.board))])

    # Check for patterns in the row
        for i in range(len(row) - 2):
            pattern = row[i:i+3]
            pattern_tuple = (pattern,)
            if pattern_tuple in self.pattern_dict:
                weight += self.pattern_dict[pattern_tuple]

    # Check for patterns in the column
        for i in range(len(col) - 2):
            pattern = col[i:i+3]
            pattern_tuple = (pattern,)
            if pattern_tuple in self.pattern_dict:
                weight += self.pattern_dict[pattern_tuple]

    # Additional heuristics: balance and blocking
        row_player = row.count('P')
        row_opponent = row.count('O')
        col_player = col.count('P')
        col_opponent = col.count('O')

    # Penalize imbalance
        weight -= abs(row_player - row_opponent) * 5
        weight -= abs(col_player - col_opponent) * 5

    # Reward moves that block opponent triples
        if 'OO.' in row or '.OO' in row or 'O.O' in row:
            weight += 50
        if 'OO.' in col or '.OO' in col or 'O.O' in col:
            weight += 50

    # Reward moves that lead to player triples
        if 'PP.' in row or '.PP' in row or 'P.P' in row:
            weight += 30
        if 'PP.' in col or '.PP' in col or 'P.P' in col:
            weight += 30

    # Remove the number after evaluation
        self.board[y][x] = None

        return weight

# MCTSNode class with Pattern-Based Enhancements
class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.move = move

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_legal_moves())

    def best_child(self, exploration_weight=1.41):
        choices_weights = [
            (child.value / (child.visits + 1e-6)) + exploration_weight * math.sqrt((2 * math.log(self.visits)) / (child.visits + 1e-6))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        untried_moves = [move for move in self.state.get_legal_moves() if move not in [child.move for child in self.children]]
        weighted_moves = []

        # Assign pattern-based weights to the untried moves
        for move in untried_moves:
            x, y, num = int(move[0]), int(move[1]), int(move[2])
            weight = self.state.calculate_move_weight(x, y, num)
            weighted_moves.append((move, weight))

        # Sort moves based on weights (higher weight moves first)
        weighted_moves.sort(key=lambda mw: mw[1], reverse=True)

        # Choose the highest weighted move
        best_move = weighted_moves[0][0]
        new_state = self.state.copy()
        new_state.perform_move(best_move)
        child_node = MCTSNode(new_state, parent=self, move=best_move)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(-result)

# MCTS with Pattern-Based Rollouts
def mcts(initial_state, itermax, exploration_weight=1.41):
    root = MCTSNode(initial_state)
    start_time = time.time()

    for _ in range(itermax):
        # Check if time limit is reached
        if time.time() - start_time >= initial_state.max_genmove_time - 1:  # Leave a 1-second buffer
            break

        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child(exploration_weight)

        if not node.is_fully_expanded():
            node = node.expand()

        result = rollout(node.state)
        node.backpropagate(result)

    best_child = root.best_child(0)
    if best_child:
        return best_child.move
    else:
        return None

def rollout(state):
    current_state = state.copy()
    while not current_state.is_terminal():
        # Calculate weights of moves based on patterns
        weighted_moves = []
        moves = current_state.get_legal_moves()
        for move in moves:
            x, y, num = int(move[0]), int(move[1]), int(move[2])
            weight = current_state.calculate_move_weight(x, y, num)
            weighted_moves.append((move, weight))

        # Normalize weights to probabilities
        total_weight = sum(weight for _, weight in weighted_moves)
        if total_weight > 0:
            probabilities = [weight / total_weight for _, weight in weighted_moves]
        else:
            probabilities = [1 / len(weighted_moves)] * len(weighted_moves)

        # Select move based on weighted probabilities
        selected_move = random.choices([move for move, _ in weighted_moves], probabilities)[0]
        current_state.perform_move(selected_move)

    return evaluate_winner(current_state)

def evaluate_winner(state):
    if state.get_legal_moves():
        return 0  # Game not finished
    return 1 if state.player == 2 else -1  # Current player loses

# ===============================================================================================
# ɅɅɅɅɅɅɅɅɅɅ End of Assignment 4 functions. ɅɅɅɅɅɅɅɅɅɅ
# ===============================================================================================

if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()

