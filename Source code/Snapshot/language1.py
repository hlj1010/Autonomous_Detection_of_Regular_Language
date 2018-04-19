from random import randrange

from automata.fa.dfa import DFA
from automata.shared.exceptions import RejectionError

import struct
import time

import subprocess
import sys
import os

dfa_0 = DFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'0', '1'},
    transitions={
        'q0': {'0': 'q0', '1': 'q1'},
        'q1': {'0': 'q0', '1': 'q2'},
        'q2': {'0': 'q2', '1': 'q1'}
    },
    initial_state='q0',
    final_states={'q1'}
)

dfa_1 = DFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'0', '1'},
    transitions={
        'q0': {'0': 'q0', '1': 'q1'},
        'q1': {'0': 'q0', '1': 'q2'},
        'q2': {'0': 'q2', '1': 'q1'}
    },
    initial_state='q0',
    final_states={'q2'}
)

dfa_2 = DFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'0', '1'},
    transitions={
        'q0': {'1': 'q0', '0': 'q1'},
        'q1': {'1': 'q0', '0': 'q2'},
        'q2': {'1': 'q2', '0': 'q1'}
    },
    initial_state='q0',
    final_states={'q1'}
)

dfa_3 = DFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'0', '1'},
    transitions={
        'q0': {'1': 'q0', '0': 'q1'},
        'q1': {'1': 'q0', '0': 'q2'},
        'q2': {'1': 'q2', '0': 'q1'}
    },
    initial_state='q0',
    final_states={'q2'}
)

dfa_4 = DFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'0', '1'},
    transitions={
        'q0': {'1': 'q0', '0': 'q1'},
        'q1': {'0': 'q2', '1': 'q0'},
        'q2': {'0': 'q1', '1': 'q2'}
    },
    initial_state='q0',
    final_states={'q1'}
)

dfa_5 = DFA(
    states={'q0', 'q1', 'q2', 'q3', 'q4'},
    input_symbols={'0', '1'},
    transitions={
        'q0': {'1': 'q0', '0': 'q1'},
        'q1': {'1': 'q2', '0': 'q1'},      # NEED TO UPDATE THIS in the drawing
        'q2': {'0': 'q1', '1': 'q3'},
        'q3': {'0': 'q1', '1': 'q0'},
        'q4': {'0': 'q4', '1': 'q4'}
    },
    initial_state='q0',
    final_states={'q2'}
)

dfa_6 = DFA(
    states={'q0', 'q1', 'q2', 'q3', 'q4'},
    input_symbols={'0', '1'},
    transitions={
        'q0': {'1': 'q0', '0': 'q1'},        # NEED TO UPDATE THIS in the drawing
        'q1': {'1': 'q2', '0': 'q1'},
        'q2': {'0': 'q1', '1': 'q3'},
        'q3': {'0': 'q1', '1': 'q0'},
        'q4': {'0': 'q4', '1': 'q4'}
    },
    initial_state='q0',
    final_states={'q3'}
)


# def generate_string(min_length, max_length, dfa, final_results, prefix):
#     retval = ''
#     current_state = dfa.initial_state

#     # for i in range(length):
#     #     transition = str(randrange(0, len(dfa.transitions[current_state])))
#     #     current_state = dfa.transitions[current_state][transition]
#     #     retval += transition

#     retry = True
#     while retry is True:
#         retry = False
#         rand_length = randrange(min_length, max_length)
#         # print("rand_length: {}".format(rand_length))
#         while ((len(retval) < rand_length) or (current_state not in dfa.final_states)) and ((prefix, retval) not in final_results):
#             transition = str(randrange(0, len(dfa.transitions[current_state])))
#             current_state = dfa.transitions[current_state][transition]
#             retval += transition
#             # print(retval)

#             if (len(retval) > (max_length * 1.25)):
#                 print("ESCAPING")
#                 break

#         try:
#             dfa.validate_input(retval)
#         except RejectionError as e:
#             retry = True
#             print("REJECTED: {}".format(e))

#     return retval


def write_results(width, final_results):

    fd0 = open('{}_dfa_0.bin'.format(width), 'ba')
    fd1 = open('{}_dfa_1.bin'.format(width), 'ba')
    # fd2 = open('{}_dfa_2.bin'.format(width), 'ba')
    # fd3 = open('{}_dfa_3.bin'.format(width), 'ba')
    # fd4 = open('{}_dfa_4.bin'.format(width), 'ba')
    # fd5 = open('{}_dfa_5.bin'.format(width), 'ba')
    # fd6 = open('{}_dfa_6.bin'.format(width), 'ba')

    for dfa_num, input_num in final_results:
        n = struct.pack('!I', input_num)
        if dfa_num == 0:
            fd0.write(n)
        elif dfa_num == 1:
            fd1.write(n)
        elif dfa_num == 2:
            fd2.write(n)
        elif dfa_num == 3:
            fd3.write(n)
        elif dfa_num == 4:
            fd4.write(n)
        elif dfa_num == 5:
            fd5.write(n)
        elif dfa_num == 6:
            fd6.write(n)

    fd0.close()
    fd1.close()
    # fd2.close()
    # fd3.close()
    # fd4.close()
    # fd5.close()
    # fd6.close()


if __name__ == "__main__":
    # s = r_generate_string(500, dfa.initial_state)

    final_results = []
    dfas = [dfa_0]  # , dfa_1, dfa_2, dfa_3, dfa_4, dfa_5, dfa_6]

    final_results = []

    if len(sys.argv) < 2:
        exit()

    if len(sys.argv) == 2:
        sets = int(sys.argv[1])

        print(os.path.basename(__file__))

        for j in range(4, 33):
            for i in range(len(dfas)):
                fd = open('{}_dfa_{}.bin'.format(j, i), 'wb')
                fd.close()

            procs = []

            for i in range(sets):
                p = subprocess.Popen(['python', os.path.basename(
                    __file__), str(i), str(sets), str(j), 'child'])

                procs.append(p)
            
            for p in procs:
                p.wait()

    elif sys.argv[4] == 'child':

        group = int(sys.argv[1])
        sets = int(sys.argv[2])
        WIDTH = int(sys.argv[3])

        start_time = time.time()
        print('{}: G{}: {} - {}'.format(WIDTH, group, int((((2**WIDTH) / sets)
                                                * group) + 1), int(((2**WIDTH) / sets) * (group + 1))))

        previous_i = int((((2**WIDTH) / sets) * group) + 1)
        for i in range(int((((2**WIDTH) / sets) * group) + 1), int(((2**WIDTH) / sets) * (group + 1))):
            input = ('{:0' + str(WIDTH) + 'b}').format(i)
            # print(input)
            interim_results = []
            match_count = 0
            for j in range(0, len(dfas)):
                try:
                    dfas[j].validate_input(input)
                    match_count += 1
                    interim_results.append((j, i))
                    # print("{}: Accepted by DFA {}!".format(i, j))

                except RejectionError as e:
                    interim_results.append((1, i))
                    # print("Rejected by DFA {}: {}".format(j, e))
                    pass

            if (len(interim_results) == 1):
                # print("\n{}: ONLY ACCEPTED BY ONE DFA!!!!!!!!!!!!\n".format(i))
                final_results.append(interim_results[0])

            # if match_count > 1:
            #     print("error")

            if len(final_results) > 100000:
                write_results(WIDTH, final_results)
                final_results = []
                print('{}: G{}: {:08x}: {:.0%} complete.  Elapsed time: {:.04}s: Evaluated {} sequences.'.format(
                    WIDTH, group, i, i/2**WIDTH, time.time() - start_time, i - previous_i))
                previous_i = i
                start_time = time.time()

        write_results(WIDTH, final_results)
