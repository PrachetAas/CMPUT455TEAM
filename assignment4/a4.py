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

    def simulate_game(self, board, player):
        """Simulate a random game from the current position."""
        board = copy.deepcopy(board)
        current_player = player

        while True:
            moves = []
            for y in range(len(board)):
                for x in range(len(board[0])):
                    for num in range(2):
                        if self.is_legal(x, y, num)[0]:
                            moves.append((x, y, num))
            
            if not moves:
                return 2 if current_player == 1 else 1  
                
            x, y, num = random.choice(moves)
            board[y][x] = num
            current_player = 3 - current_player  

    def mcts_search(self, time_limit):
        """Perform MCTS search from the current position."""
        start_time = time.time()
        
        states = {}
        
        while time.time() - start_time < time_limit:
            
            board = copy.deepcopy(self.board)
            player = self.player
            visited_states = []
            
           
            while True:

                state = (tuple(tuple(row) for row in board), player)
                visited_states.append(state)
                
                legal_moves = []
                for y in range(len(board)):
                    for x in range(len(board[0])):
                        for num in range(2):
                            if self.is_legal(x, y, num)[0]:
                                legal_moves.append((x, y, num))
                
                if not legal_moves:
                    break
                
                # If state is new, initialize it and break for simulation
                if state not in states:
                    states[state] = (0, 0)
                    break
                
                # Select move using UCT
                total_visits = states[state][1]
                selected_move = None
                best_value = float('-inf')
                
                random.shuffle(legal_moves)  # Add randomization to move selection
                
                for move in legal_moves:
                    x, y, num = move
                    temp_board = copy.deepcopy(board)
                    temp_board[y][x] = num
                    next_state = (tuple(tuple(row) for row in temp_board), 3 - player)
                    
                    if next_state not in states:
                        selected_move = move
                        break
                    
                    wins, visits = states[next_state]
                    uct = self.uct_value(total_visits, wins, visits)
                    if uct > best_value:
                        best_value = uct
                        selected_move = move
                
                # Make the selected move
                x, y, num = selected_move
                board[y][x] = num
                player = 3 - player  # Switch player
            
            # Simulation
            winner = self.simulate_game(board, player)
            
            # Backpropagation
            for state in visited_states:
                if state not in states:
                    states[state] = (0, 0)
                wins, visits = states[state]
                
                # Update wins and visits
                visits += 1
                if winner != state[1]:  # If the player who didn't make the move from this state won
                    wins += 1
                states[state] = (wins, visits)
        
        # Select best move from root state
        
        legal_moves = []
        best_visits = -1
        best_move = None
        
        for y in range(len(self.board)):
            for x in range(len(self.board[0])):
                for num in range(2):
                    if self.is_legal(x, y, num)[0]:
                        temp_board = copy.deepcopy(self.board)
                        temp_board[y][x] = num
                        next_state = (tuple(tuple(row) for row in temp_board), 3 - self.player)
                        
                        if next_state in states:
                            visits = states[next_state][1]
                            if visits > best_visits:
                                best_visits = visits
                                best_move = (x, y, num)
        
        return best_move

    def genmove(self, args):
        """Generate a move using MCTS."""
        try:
            signal.alarm(self.max_genmove_time)
            
            # Check if there are any legal moves
            moves = self.get_legal_moves()
            if not moves:
                print("resign")
                return True
            
            # Run MCTS with slightly less time than the limit to account for overhead
            best_move = self.mcts_search(self.max_genmove_time - 0.1)
            
            if best_move is None:
                # Fallback to random move if MCTS fails
                moves = self.get_legal_moves()
                rand_move = moves[random.randint(0, len(moves)-1)]
                self.play(rand_move)
                print(" ".join(rand_move))
            else:
                # Play the MCTS move
                move = [str(x) for x in best_move]
                self.play(move)
                print(" ".join(move))
            
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