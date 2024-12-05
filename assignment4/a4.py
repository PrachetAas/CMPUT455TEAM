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
            "set_opponent": self.set_opponent,
            "set_student_as_player": self.set_student_as_player,
            "play_game": self.play_game
        }
        self.board = [[None]]
        self.player = 1
        self.max_genmove_time = 1
        self.transposition_table = {}
        self.zobrist_table = {}
        self.hash_value = 0
        self.eval_cache = {}
        self.start_time = 0
        self.history = []
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
        new_interface.zobrist_table = self.zobrist_table.copy()
        new_interface.hash_value = self.hash_value
        new_interface.eval_cache = self.eval_cache.copy()
        new_interface.history = self.history.copy()
        new_interface.start_time = self.start_time
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
        # Uncomment the following line for debugging purposes
        # print(f"Board after play {args}:\n{self.board_to_string()}", file=sys.stderr)
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
            if current_player == self.student_player_num:
                # Your player's turn
                move = self.genmove([])
                if move == "resign":
                    opponent_process.stdin.write("resign\n")
                    opponent_process.stdin.flush()
                    opponent_response = self.read_opponent_response(opponent_process)
                    winner = 3 - self.student_player_num
                    print(winner)
                    game_over = True
                else:
                    # Apply your move
                    self.play(move.strip().split())
                    # Send your move to the opponent
                    opponent_process.stdin.write(f"play {move}\n")
                    opponent_process.stdin.flush()
                    opponent_response = self.read_opponent_response(opponent_process)
            else:
                # Opponent's turn
                opponent_process.stdin.write("genmove\n")
                opponent_process.stdin.flush()
                opponent_move = self.read_opponent_response(opponent_process)
                if opponent_move == "resign":
                    print(self.student_player_num)
                    game_over = True
                else:
                    # Apply opponent's move
                    self.play(opponent_move.strip().split())
                    # Send the move back to the opponent to keep game states in sync
                    opponent_process.stdin.write(f"play {opponent_move}\n")
                    opponent_process.stdin.flush()
                    opponent_response = self.read_opponent_response(opponent_process)

            # Check for game over
            if self.is_terminal():
                winner = current_player
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
        return move

    # MCTS Integration with Simplified Rollouts
    def genmove(self, args):
        self.start_time = time.time()
        signal.alarm(int(self.max_genmove_time - 2))  # Set the alarm, 2-second safety margin
        try:
            if len(self.get_legal_moves()) == 0:
                print("resign")
                return "resign"

            # Run MCTS to select the best move
            itermax = 10000  # Adjust the number of iterations as appropriate
            best_move = mcts(self.copy(), itermax)

            if best_move:
                self.perform_move(best_move)
                move_str = " ".join(best_move)
                print(move_str)
                return move_str
            else:
                # If MCTS didn't find a move, pick a random move
                moves = self.get_legal_moves()
                if moves:
                    rand_move = moves[random.randint(0, len(moves) - 1)]
                    self.perform_move(rand_move)
                    move_str = " ".join(rand_move)
                    print(move_str)
                    return move_str
                else:
                    print("resign")
                    return "resign"

        except TimeoutException:
            print("resign")
            return "resign"
        except Exception as e:
            print(f"Exception in genmove: {e}", file=sys.stderr)
            print("resign")
            return "resign"
        finally:
            signal.alarm(0)  # Cancel the alarm

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

    # ===============================================================================================
    # ɅɅɅɅɅɅɅɅɅɅ End of Assignment 4 functions. ɅɅɅɅɅɅɅɅɅɅ
    # ===============================================================================================

# MCTSNode class with Simplified Rollouts
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

    def best_child(self, exploration_weight=math.sqrt(2)):
        choices_weights = [
            (child.value / (child.visits + 1e-6)) + exploration_weight * math.sqrt((2 * math.log(self.visits + 1e-6)) / (child.visits + 1e-6))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self):
        untried_moves = [move for move in self.state.get_legal_moves() if move not in [child.move for child in self.children]]
        if not untried_moves:
            return None

        # Randomly select a move to expand
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

# MCTS with Random Rollouts
def mcts(initial_state, itermax, exploration_weight=math.sqrt(2)):
    root = MCTSNode(initial_state)

    while True:
        # Check if time limit is reached
        if time.time() - initial_state.start_time >= initial_state.max_genmove_time - 2:
            break

        node = root
        while node.is_fully_expanded() and node.children:
            node = node.best_child(exploration_weight)
            # Time check
            if time.time() - initial_state.start_time >= initial_state.max_genmove_time - 2:
                break

        if not node.is_fully_expanded():
            node = node.expand()
            # Time check
            if time.time() - initial_state.start_time >= initial_state.max_genmove_time - 2:
                break

        if node is None:
            break

        result = rollout(node.state)
        node.backpropagate(result)
        # Time check
        if time.time() - initial_state.start_time >= initial_state.max_genmove_time - 2:
            break

    best_child = root.best_child(0)
    if best_child:
        return best_child.move
    else:
        return None

def rollout(state):
    current_state = state.copy()
    while not current_state.is_terminal():
        # Time check
        if time.time() - state.start_time >= state.max_genmove_time - 2:
            break

        moves = current_state.get_legal_moves()
        if not moves:
            break

        selected_move = random.choice(moves)
        current_state.perform_move(selected_move)

    return evaluate_winner(current_state)

def evaluate_winner(state):
    if state.is_terminal():
        return 1 if state.player != 1 else -1
    return 0

if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()


