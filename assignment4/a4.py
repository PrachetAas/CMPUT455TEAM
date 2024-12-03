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


    def initialize_zobrist_hash(self):
        row_range = len(self.board)
        col_range = len(self.board[0])
        for x in range(row_range):
            for y in range(col_range):
                self.zobrist_table[(x, y)] = [random.getrandbits(64) for _ in range(2)]

    def update_zobrist_hash(self, x, y, piece):
        self.hash_value ^= self.zobrist_table[(x, y)][piece]

    def genmove(self, args):
        self.start_time = time.time()
        try:
            moves = self.get_legal_moves()
            if len(moves) == 0:
                print("resign")
                return True

            best_move = mcts(self, itermax=1000)  # Adjust itermax for more simulations

            if best_move:
                self.perform_move(best_move)
                print(" ".join(best_move))
            else:
                rand_move = moves[random.randint(0, len(moves) - 1)]
                self.perform_move(rand_move)
                print(" ".join(rand_move))

        except TimeoutException:
            print("resign")
        return True

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

    def evaluate_balance(self, zeros, ones, total_length):
        balance = abs(zeros - ones)
        max_balance = total_length // 2
        return (max_balance - balance) ** 2

    

    

    def perform_move(self, move):
        x, y, num = int(move[0]), int(move[1]), int(move[2])
        self.board[y][x] = num
        self.update_zobrist_hash(x, y, num)
        self.history.append((y, x, num))
        self.player = 2 if self.player == 1 else 1

    def is_terminal(self):
        return len(self.get_legal_moves()) == 0

    def get_result(self, player_num):
        if self.is_terminal():
            return -1 if self.player == player_num else 1
        return 0

# Monte Carlo Tree Search Implementation
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
        move = random.choice(untried_moves)
        new_state = self.state.copy()
        new_state.perform_move(move)
        child_node = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(-result)

def mcts(initial_state, itermax, exploration_weight=1.41):
    root = MCTSNode(initial_state)

    for _ in range(itermax):
        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child(exploration_weight)

        if not node.is_fully_expanded():
            node = node.expand()

        result = rollout(node.state)
        node.backpropagate(result)

    return root.best_child(0).move

def rollout(state):
    current_state = state.copy()
    while current_state.get_legal_moves():
        move = random.choice(current_state.get_legal_moves())
        current_state.perform_move(move)
    return evaluate_winner(current_state)

def evaluate_winner(state):
    if state.get_legal_moves():
        return 0
    return 1 if state.player == 2 else -1

    # ===============================================================================================
    # ɅɅɅɅɅɅɅɅɅɅ End of Assignment 4 functions. ɅɅɅɅɅɅɅɅɅɅ
    # ===============================================================================================


if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()
