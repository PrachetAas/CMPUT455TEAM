# CMPUT 455 Assignment 4 starter code
# Implement the specified commands to complete the assignment
# Full assignment specification here: https://webdocs.cs.ualberta.ca/~mmueller/courses/cmput455/assignments/a4.html

import sys
import random
import signal
import math
import copy
import time

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
            "help" : self.help,
            "game" : self.game,
            "show" : self.show,
            "play" : self.play,
            "legal" : self.legal,
            "genmove" : self.genmove,
            "winner" : self.winner,
            "timelimit": self.timelimit
        }
        self.board = [[None]]
        self.player = 1
        self.max_genmove_time = 1
        signal.signal(signal.SIGALRM, handle_alarm)

        self.pattern = {
            ('X1.0X', 0): 21,  # Very high weight for 0
            ('1...X', 0): 21,
            ('11.0X', 0): 21,
            ('X0.XX', 0): 21,
            ('1...0', 0): 21,
            ('.0.0.', 0): 18,  # Good balanced patterns
            ('.0.0.', 1): 18,
            ('01.1.', 0): 18,
            ('.0.XX', 0): 18,
            ('0...X', 0): 18,
            ('10..1', 0): 20,
            ('10..1', 1): 12,
            ('00.00', 0): 20,
            ('....X', 1): 20,
            ('10...', 0): 20,
            ('X...X', 1): 20,
            ('10.00', 1): 20,
            ('X..XX', 0): 20,
            ('.0.1X', 0): 20,
        }

        self.pattern_rotate = {}
        for (pattern, num), weight in self.pattern.items():
            self.add_rotations(pattern, num, weight)
    
    #====================================================================================================================
    # VVVVVVVVVV Start of predefined functions. You may modify, but make sure not to break the functionality. VVVVVVVVVV
    #====================================================================================================================

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
                print("Argument '" + arg + "' cannot be interpreted as a number.\nExpected arguments:", template, file=sys.stderr)
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
            self.board.append([None]*n)
        self.player = 1
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
        if  x >= 0 and x < len(self.board[0]) and\
                y >= 0 and y < len(self.board) and\
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
        if  x < 0 or x >= len(self.board[0]) or y < 0 or y >= len(self.board):
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

    #===============================================================================================
    # ɅɅɅɅɅɅɅɅɅɅ End of predefined functions. ɅɅɅɅɅɅɅɅɅɅ
    #========================================================================= ======================

    #===============================================================================================
    # VVVVVVVVVV Start of Assignment 4 functions. Add/modify as needed. VVVVVVVV
    #===============================================================================================

    def uct_value(self, total_visits, node_wins, node_visits):
        """Calculate the UCT value for a node."""
        if node_visits == 0:
            return float('inf')
        return (node_wins / node_visits) + 1.414 * math.sqrt(math.log(total_visits) / node_visits)

    def simulate_random_game(self, current_board, current_player):
        """Simulate a game using pattern weights to guide move selection."""
        board = copy.deepcopy(current_board)
        player = current_player
        
        while True:
            moves = []
            total_weight = 0
            for y in range(len(board)):
                for x in range(len(board[0])):
                    for num in range(2):
                        if self.valid_move(x, y, num):
                            weight = self.calculate_move_weight(x, y, num)
                            moves.append((x, y, num))
                            total_weight += weight
            
            if not moves:
                return 3 - player
            
            # Use weighted random selection instead of uniform random
            weights = [self.calculate_move_weight(x, y, num) for x, y, num in moves]
            x, y, num = random.choices(moves, weights=weights)[0]
            board[y][x] = num
            player = 3 - player

    def evaluate_move(self, move, simulations=10):
        """Evaluate a move by running multiple random simulations."""
        x, y, num = move
        wins = 0
        
        # Create a copy of the current state
        temp_board = copy.deepcopy(self.board)
        temp_board[y][x] = num
        next_player = 3 - self.player
        
        # Run simulations from the resulting position
        for _ in range(simulations):
            winner = self.simulate_random_game(temp_board, next_player)
            if winner != next_player:  # If the opponent (next_player) loses
                wins += 1
                
        return wins / simulations
    
    def add_rotations(self, pattern, num, weight):
        """Adds all rotations (0, 90, 180, 270 degrees) of the pattern."""
        rotations = [pattern, self.rotate_pattern(pattern, 90),
                     self.rotate_pattern(pattern, 180), self.rotate_pattern(pattern, 270)]
        for rotated_pattern in rotations:
            self.pattern_rotate[(rotated_pattern, num)] = weight

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
        row_weight = self.pattern_rotate.get((row_pattern, num), 10)  # Default weight if no match
        col_weight = self.pattern_rotate.get((col_pattern, num), 10)
        return row_weight + col_weight

    def genmove(self, args):
        try:
            signal.alarm(self.max_genmove_time)
            
            # Get all legal moves with their weights
            moves = []
            total_weight = 0
            for y in range(len(self.board)):
                for x in range(len(self.board[0])):
                    for num in range(2):
                        if self.valid_move(x, y, num):
                            weight = self.calculate_move_weight(x, y, num)
                            moves.append((x, y, num, weight))
                            total_weight += weight
            
            if not moves:
                print("resign")
                return True
            
            # Convert weights to probabilities and evaluate moves
            weighted_moves = [(x, y, num, weight/total_weight) for x, y, num, weight in moves]
            weighted_moves.sort(key=lambda x: x[3], reverse=True)  # Sort by probability
            
            # Evaluate the most promising moves first
            best_move = None
            best_score = -1
            start_time = time.time()
            
            for x, y, num, prob in weighted_moves:
                if time.time() - start_time > self.max_genmove_time - 0.5:
                    break
                    
                score = self.evaluate_move((x, y, num))
                weighted_score = score + prob # Combine MCTS score with pattern weight
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_move = (x, y, num)
            
            # Convert move to string format and play it
            move_str = [str(best_move[0]), str(best_move[1]), str(best_move[2])]
            self.play(move_str)
            print(" ".join(move_str))
            
            signal.alarm(0)
            
        except TimeoutException:
            print("resign")
            
        return True
    
    #===============================================================================================
    # ɅɅɅɅɅɅɅɅɅɅ End of Assignment 4 functions. ɅɅɅɅɅɅɅɅɅɅ
    #===============================================================================================
    
if __name__ == "__main__":
    interface = CommandInterface()
    interface.main_loop()