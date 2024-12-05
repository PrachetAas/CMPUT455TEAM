# CMPUT 455 Assignment 4 starter code
# Implement the specified commands to complete the assignment
# Full assignment specification here: https://webdocs.cs.ualberta.ca/~mmueller/courses/cmput455/assignments/a4.html

import sys
import random
import signal
import time
import math
import numpy as np


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
            "timelimit": self.timelimit
        }
        self.board = [[None]]
        self.player = 1
        self.max_genmove_time = 1
        self.pattern_dict = {}

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
            print("? Uknown command.\nType 'help' to list known commands.", file=sys.stderr)
            print("= -1\n")
            return False
        try:
            return self.command_dict[command](args)
        except Exception as e:
            print("Command '" + str + "' failed with exception:", file=sys.stderr)
            print(e, file=sys.stderr)
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
    
        return new_interface

    # Will continuously receive and execute commands
    # Commands should return True on success, and False on failure
    # Every command will print '= 1' or '= -1' at the end of execution to indicate success or failure respectively
    def main_loop(self):
        while True:
            str = input()
            if str.split(" ")[0] == "exit":
                print("= 1\n")
                return True
            if self.process_command(str):
                print("= 1\n")

    # Will make sure there are enough arguments, and that they are valid numbers
    # Not necessary for commands without arguments
    def arg_check(self, args, template):
        converted_args = []
        if len(args) < len(template.split(" ")):
            print("Not enough arguments.\nExpected arguments:", template, file=sys.stderr)
            print("Recieved arguments: ", end="", file=sys.stderr)
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
        self.update_zobrist_hash(y, x, num)
        self.history.append((y, x, num))
        if self.player == 1:
            self.player = 2
        else:
            self.player = 1
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


    def rotate_pattern(self, pattern, angle):
        """Rotate pattern by the specified angle (90, 180, 270 degrees)."""
        if angle == 180:
            return pattern[::-1]  # Simple reverse for 180°
        if angle == 90 or angle == 270:
            return ''.join([pattern[i] for i in (4, 2, 0, 3, 1)])  # Rotations for 90° and 270°
        return pattern

    def get_board_pattern(self, x, y, direction):
        """Helper function to extract a row or column pattern based on the direction (row or column)."""
        n, m = len(self.board[0]), len(self.board)
        pattern = ""
        if direction == "row":
            for i in range(x - 2, x + 3):
                if 0 <= i < n:
                    pattern += "." if self.board[y][i] is None else str(self.board[y][i])
                else:
                    pattern += "X"
        elif direction == "column":
            for j in range(y - 2, y + 3):
                if 0 <= j < m:
                    pattern += "." if self.board[j][x] is None else str(self.board[j][x])
                else:
                    pattern += "X"
        return pattern

    def calculate_move_weight(self, x, y, num):
        """Calculate the weight of a move at (x, y) for 'num' (0 or 1)."""
        row_pattern = self.get_board_pattern(x, y, "row")
        col_pattern = self.get_board_pattern(x, y, "column")
        
        # Evaluate the board with pattern rotations
        weights = []
        for angle in [0, 90, 180, 270]:
            rotated_row_pattern = self.rotate_pattern(row_pattern, angle)
            rotated_col_pattern = self.rotate_pattern(col_pattern, angle)
            # Higher weight if the move creates a favorable pattern
            row_weight = self.pattern_dict.get((rotated_row_pattern, num), 10)
            col_weight = self.pattern_dict.get((rotated_col_pattern, num), 10)
            weights.append(row_weight + col_weight)
        
        return max(weights)  # Return the highest weight as an evaluation metric

    # Evaluate Balance Function
    def evaluate_balance(self, zeros, ones, total_length):
        balance = abs(zeros - ones)
        max_balance = total_length // 2
        return (max_balance - balance) ** 2

    # Board Evaluation Function
    def evaluate_board(self):
        # Using Zobrist hashing to cache the board evaluations for quicker lookup
        if self.hash_value in self.eval_cache:
            return self.eval_cache[self.hash_value]

        score = 0

        # Evaluate rows for balance, three-in-a-row, and control
        for row in self.board:
            zeros = sum(1 for x in row if x == 0)
            ones = sum(1 for x in row if x == 1)
            score += self.evaluate_balance(zeros, ones, len(row))

            # Penalize three consecutive same values
            for i in range(len(row) - 2):
                if row[i] == row[i + 1] == row[i + 2] and row[i] is not None:
                    score -= 5 * (len(self.board) - i)

        # Evaluate columns similarly
        for col in range(len(self.board[0])):
            zeros = ones = 0
            for row in range(len(self.board)):
                if self.board[row][col] == 0:
                    zeros += 1
                elif self.board[row][col] == 1:
                    ones += 1

            score += self.evaluate_balance(zeros, ones, len(self.board))

            # Penalize three consecutive same values in columns
            for i in range(len(self.board) - 2):
                if self.board[i][col] == self.board[i + 1][col] == self.board[i + 2][col] and self.board[i][col] is not None:
                    score -= 5 * (len(self.board) - i)

        # Reward center control
        center_x, center_y = len(self.board[0]) // 2, len(self.board) // 2
        if self.board[center_y][center_x] is not None:
            score += 10

        # Add heuristic for potential two-in-a-row
        for row in self.board:
            for i in range(len(row) - 1):
                if row[i] == row[i + 1] and row[i] is not None:
                    score += 3

        # Invert score if it's the opponent's turn
        if self.player != 1:
            score = -score

        self.eval_cache[self.hash_value] = score
        return score

    # MCTS Integration with Pattern-Based Enhancements
    def genmove(self, args):
        self.start_time = time.time()
        try:
            moves = self.get_legal_moves()
            if len(moves) == 0:
                print("resign")
                return "resign"

            best_move = mcts(self, itermax=10000)  # Adjust itermax as needed

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
        for x in range(row_range):
            for y in range(col_range):
                self.zobrist_table[(x, y)] = [random.getrandbits(64) for _ in range(2)]

    def update_zobrist_hash(self, x, y, piece):
        self.hash_value ^= self.zobrist_table[(x, y)][piece]

    def perform_move(self, move):
        x, y, num = int(move[0]), int(move[1]), int(move[2])
        self.board[y][x] = num
        self.update_zobrist_hash(x, y, num)
        self.history.append((y, x, num))
        self.player = 2 if self.player == 1 else 1

    # Terminal Check Function
    def is_terminal(self):
        return len(self.get_legal_moves()) == 0

    # Get Result Function
    def get_result(self, player_num):
        if self.is_terminal():
            return -1 if self.player == player_num else 1
        return 0

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
    while not current_state.is_terminal():  # Stop rolling out if terminal state
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
