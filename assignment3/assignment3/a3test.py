# CMPUT 455 assignment 3 testing script
# Run using: python3 a3test.py a3.py assignment3-public-tests.txt [-v]

import subprocess
import sys
import time
import signal
import os
import re
import math

# Default maximum command execution time in seconds
TIMEOUT = 1
DYNAMIC_TIMEOUT = 1

USE_COLOR = True

if USE_COLOR:
    # Color codes
    RED = "\033[31m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    RESET = "\033[0m"
else:
    RED = ""
    GREEN = ""
    BLUE = ""
    RESET = ""

# Functions necessary for checking timeouts
class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException("Function timed out.")

# Class for specifying tests and recording results
class Test:
    def __init__(self, command, expected, id, to_mark):
        self.command = command
        self.expected = expected
        self.id = id
        self.received = ""
        self.passed = None
        self.matched = None
        self.to_mark = to_mark
        self.notes = ""

    def to_dict(self):
        return {"command": self.command,\
                "expected": self.expected,\
                "id": self.id,\
                "received": self.received,\
                "passed": self.passed,\
                "matched": self.matched,\
                "notes": self.notes}

    # Printed representation of a test
    def __str__(self):
        s = " === Test "+ str(self.id) +" === \nCommand: '" + self.command + "'\nExpected: "
        
        if "\n" in self.expected[:-1]:
            s += "\n'\n" + self.expected + "'\n"
        else:
            s += "'" + self.expected.strip() + "'\n"

        s += "Received: "
        if "\n" in self.received.strip():
            s += "\n'\n"
        else:
            s += "'"

        if self.matched:
            s += f"{GREEN}" + self.received
        else:
            matching = True
            s += f"{GREEN}"
            for i in range(len(self.received.strip())):
                if i < len(self.expected) and self.expected[i] == self.received[i]:
                    if not matching:
                        s += f"{RESET}" + f"{GREEN}"
                else:
                    if matching:
                        s += f"{RESET}" + f"{RED}"
                s += self.received[i]

        if "\n" not in self.received.strip():
            s = s.strip()
        s += f"{RESET}'\n"

        if self.passed and self.matched:
            s += f"{GREEN}+ Success{RESET}\n"
        
        if not self.passed:
            s += f"{RED}- This command failed with error:\n'" + self.notes + f"'{RESET}\n"

        if self.to_mark:
            s += "This command will be marked.\n"
        else:
            s += "This command will NOT be marked.\n"

        return s.strip()+"\n"
    
# Convert a test file into test objects
def file_to_tests(file_name):
    test_lines = []
    with open(file_name, "r") as tf:
        test_lines = tf.readlines()

    i = 0
    while i < len(test_lines):
        # Strip comments
        test_lines[i] = test_lines[i].split("#")[0].strip()
        # Delete whitespace lines
        if len(test_lines[i]) == 0:
            del test_lines[i]
        else:
            i += 1

    # Create tests
    tests = []
    i = 0
    while i < len(test_lines):
        command = test_lines[i]
        i += 1
        expected = test_lines[i] + "\n"
        while test_lines[i][0] != '=':
            i += 1
            expected += test_lines[i] + "\n"
        to_mark = False
        if command[0] == '?':
            to_mark = True
            command = command[1:]
        tests.append(Test(command, expected, len(tests)+1, to_mark))
        i += 1
    return tests

# Send a command, returns whether the command passed, the output received, and any error messages
def send_command(process, command, expected_fail = False, to_mark = False):
    global TIMEOUT, DYNAMIC_TIMEOUT
    if command.split(" ")[0] == "timelimit":
        DYNAMIC_TIMEOUT = int(command.split(" ")[1])
    try:
        process.stdin.write(command+"\n")
        process.stdin.flush()
        output = ""
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(TIMEOUT) if not to_mark else signal.alarm(TIMEOUT+DYNAMIC_TIMEOUT)
            line = process.stdout.readline()
            while line[0] != "=":
                if len(line.strip()) > 0:
                    output += line
                line = process.stdout.readline()
            signal.alarm(0)
            output += line

            if '= -1' in line and not expected_fail:
                return False, output, "Command failed with return code -1."
            else:
                return True, output, ""
        
        except TimeoutException:
            return False, output, "Command timeout, exceeded maximum allowed time of " + str(TIMEOUT) + " seconds."

    except Exception as e:
        signal.alarm(0)
        return False, "", "Process error:\n" + str(e)

def perform_test(process, test):
    test.passed, test.received, test.notes = send_command(process, test.command, expected_fail="= -1" in test.expected, to_mark=test.to_mark)
    if test.expected[0] == '@':
        exp_pattern = re.compile((test.expected.strip())[1:], re.DOTALL)
        test.matched = bool(exp_pattern.match(test.received.strip()))
    else:
        test.matched = test.expected == test.received
    return test.matched

# Test a given process on a number of tests. Prints and returns results.
def test_process(process, tests, verbose=False, print_output=False):
    t0 = time.time()
    successful = []
    failed = []
    mismatched = []
    test_num = 1
    for test in tests:
        if print_output:
            print("Test", test_num, "/", len(tests), "(" + str(round(100 * test_num / len(tests))) + "%)", end="\r")
        test_num += 1
        perform_test(process, test)
        if not test.passed:
            failed.append(test)
        elif not test.matched:
            mismatched.append(test)
        else:
            successful.append(test)        

    if print_output:
        print()
        if verbose:
            for test in tests:
                print(test)

        print(f"{BLUE}\tFailed commands (" + str(len(failed)) + f"):\n{RESET}")
        for test in failed:
            print(test)
        print(f"{BLUE}\tSuccessful commands with mismatched outputs: (" + str(len(mismatched)) + f"):\n{RESET}")
        for test in mismatched:
            print(test)
        print(f"{BLUE}\tSummary report:\n{RESET}")
        print(len(tests), "Tests performed")
        print(f"{GREEN}" + str(len(successful)) + " Successful (" + str(round(100*len(successful) / len(tests))) + f"%){RESET}")
        print(f"{RED}" + str(len(failed)) + " Failed (" + str(round(100*len(failed) / len(tests))) + f"%){RESET}")
        print(f"{RED}" + str(len(mismatched)) + " Mismatched (" + str(round(100*len(mismatched) / len(tests))) + f"%){RESET}")

        print(f"{BLUE}\tMarks report:\n{RESET}")
        passed_marked = len([x for x in successful if x.to_mark])
        all_marked = len([x for x in tests if x.to_mark])
        mark = round(math.floor(passed_marked / all_marked * 20) / 10, 1)
        if mark == 0 and passed_marked != 0:
            mark = 0.1
        print(str(passed_marked) + " / " + str(all_marked) + " marked tests = " + str(mark) + " / 2.0 marks.")
        print("\nFinished in", round(time.time() - t0, 2), "seconds.")

    return successful, failed, mismatched

def test_assignment(proc_name, test_name, verbose = False, marking = False):
    try:
        proc = subprocess.Popen(["python3", proc_name], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        time.sleep(0.1)
        if proc.poll() is not None:
            raise Exception("Program exited with return code: "+str(proc.poll()))
    except Exception as e:
        if marking:
            raise e
        print("Failed to start " + proc_name)
        return

    tests = file_to_tests(test_name)

    s, f, m = test_process(proc, tests, verbose, not marking)
    proc.terminate()
    return s, f, m


if __name__ == "__main__":
    if len(sys.argv) != 3 and (len(sys.argv) != 4 or sys.argv[3] != "-v"):
        print("Usage:\npython3 a1test.py aX.py assignmentX-public-tests.txt [-v]")
        sys.exit()

    verbose = len(sys.argv) == 4 and sys.argv[3] == "-v"
    print_results = True

    if not os.path.isfile(sys.argv[1]):
        print("File '" + sys.argv[1] + "' not found.")
        sys.exit()
    if not os.path.isfile(sys.argv[2]):
        print("File '" + sys.argv[2] + "' not found.")
        sys.exit()

    test_assignment(sys.argv[1], sys.argv[2], verbose=verbose)