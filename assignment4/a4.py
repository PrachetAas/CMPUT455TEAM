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

def handle_alarm(signum, frame):
    raise TimeoutException

class Node:
    def __init__(self, state, parent, move, player):
        self.state = state  
        self.parent = parent
        self.move = move  
        self.player = player  
        self.wins = 0
        self.visits = 0
        self.children = []
        self.untried_moves = []
        if not self.is_terminal():
            self.untried_moves = self.get_legal_moves()
    
    def is_terminal(self):
        return len(self.get_legal_moves()) == 0

    def get_legal_moves(self):
        moves = []
        for y in range(len(self.state)):
            for x in range(len(self.state[0])):
                for num in range(2):
                    legal, _ = self.is_legal(x, y, num)
                    if legal:
                        moves.append((str(x), str(y), str(num)))
        return moves

    def UCT_select_child(self):
        return max(self.children, key=lambda child: self.uct_value(child))
    
    def uct_value(self, child):
        """Calculate the UCT value for a child node."""
        if child.visits == 0:
            return float('inf')
        return (child.wins / child.visits) + 1.414 * math.sqrt(math.log(self.visits) / child.visits)
    
    def is_legal(self, x, y, num):
        if self.state[y][x] is not None:
            return False, "occupied"
        
        consecutive = 0
        count = 0
        self.state[y][x] = num
        for row in range(len(self.state)):
            if self.state[row][x] == num:
                count += 1
                consecutive += 1
                if consecutive >= 3:
                    self.state[y][x] = None
                    return False, "three in a row"
            else:
                consecutive = 0
        too_many = count > len(self.state) // 2 + len(self.state) % 2
        
        consecutive = 0
        count = 0
        for col in range(len(self.state[0])):
            if self.state[y][col] == num:
                count += 1
                consecutive += 1
                if consecutive >= 3:
                    self.state[y][x] = None
                    return False, "three in a row"
            else:
                consecutive = 0
        if too_many or count > len(self.state[0]) // 2 + len(self.state[0]) % 2:
            self.state[y][x] = None
            return False, "too many " + str(num)

        self.state[y][x] = None
        return True, ""

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

    def select_node(self, node):
        while not node.is_terminal():
            if node.untried_moves:
                return self.expand(node)
            else:
                node = node.UCT_select_child()
        return node
    
    def make_move(self, board, move):
        x, y, num = move
        new_state = copy.deepcopy(board)
        new_state[int(y)][int(x)] = num
        return new_state

    def expand(self, node):
        move = node.untried_moves.pop()
        next_state = self.make_move(node.state, move)
        child_node = Node(state=next_state, parent=node, move=move, player=3 - node.player)
        node.children.append(child_node)
        return child_node
    
    def simulate(self, node):
        current_state = copy.deepcopy(node.state)
        current_player = node.player
        while True:
            moves = self.get_legal_moves_with_state(current_state)
            if not moves:
                return 3 - current_player 
            move = self.select_move(current_state, moves, current_player)
            current_state = self.make_move(current_state, move)
            current_player = 3 - current_player
    
    def select_move(self, state, moves, player):
        # Simple heuristic: prioritize moves that prevent 'three in a row'
        best_moves = []
        for move in moves:
            x, y, num = move

            state[y][x] = num

            opponent = 3 - player
            opponent_has_three = self.check_three_in_row(state, opponent)
            state[y][x] = None
            if not opponent_has_three:
                best_moves.append(move)
        if best_moves:
            return random.choice(best_moves)
        else:
            # If all moves result in opponent's 'three in a row', pick randomly
            return random.choice(moves)
        
    def check_three_in_row(self, state, player):

        for y in range(len(state)):
            consecutive = 0
            for x in range(len(state[0])):
                if state[y][x] == player:
                    consecutive += 1
                    if consecutive >= 3:
                        return True
                else:
                    consecutive = 0
        for x in range(len(state[0])):
            consecutive = 0
            for y in range(len(state)):
                if state[y][x] == player:
                    consecutive += 1
                    if consecutive >= 3:
                        return True
                else:
                    consecutive = 0
        return False


    def get_legal_moves_with_state(self, state):
        moves = []
        for y in range(len(state)):
            for x in range(len(state[0])):
                for num in range(2):
                    legal, _ = self.is_legal_state(state, x, y, num)
                    if legal:
                        moves.append((x, y, num))
        return moves
    
    def is_legal_state(self, state, x, y, num):
    # Same as is_legal above but takes state/board param instead and uses that instead of self.board
        if state[y][x] is not None:
            return False, "occupied"
        
        consecutive = 0
        count = 0
        state[y][x] = num
        for row in range(len(state)):
            if state[row][x] == num:
                count += 1
                consecutive += 1
                if consecutive >= 3:
                    state[y][x] = None
                    return False, "three in a row"
            else:
                consecutive = 0
        too_many = count > len(state) // 2 + len(state) % 2
        
        consecutive = 0
        count = 0
        for col in range(len(state[0])):
            if state[y][col] == num:
                count += 1
                consecutive += 1
                if consecutive >= 3:
                    state[y][x] = None
                    return False, "three in a row"
            else:
                consecutive = 0
        if too_many or count > len(state[0]) // 2 + len(state[0]) % 2:
            state[y][x] = None
            return False, "too many " + str(num)
        state[y][x] = None
        return True, ""

    def backpropagate(self, node, winner):
        while node is not None:
            node.visits += 1
            if node.player == winner:
                node.wins += 1
            node = node.parent

    def genmove(self, args):
        try:
            signal.alarm(self.max_genmove_time)
            curr_board = copy.deepcopy(self.board)
            curr_node = Node(state=curr_board, parent=None, move=None, player=self.player)
            
            start_time = time.time()
            while time.time() - start_time < self.max_genmove_time - 0.5:
                # MCTS
                node = self.select_node(curr_node)
                winner = self.simulate(node)
                self.backpropagate(node, winner)
            
            best_child = max(curr_node.children, key=lambda child: child.visits)
            best_move = best_child.move
            
            self.play([str(best_move[0]), str(best_move[1]), str(best_move[2])])
            
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