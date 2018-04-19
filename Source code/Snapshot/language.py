#!/usr/bin/env python
import time
import numpy as np
from random import randrange

from automata.fa.dfa import DFA
from automata.shared.exceptions import RejectionError

dfa_0 = DFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={3, 6},
    transitions={
        'q0': {3: 'q0', 6: 'q1'},
        'q1': {3: 'q0', 6: 'q2'},
        'q2': {3: 'q2', 6: 'q1'}
    },
    initial_state='q0',
    final_states={'q1'}
)

dfa_1 = DFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'3', '6'},
    transitions={
        'q0': {'3': 'q0', '6': 'q1'},
        'q1': {'3': 'q0', '6': 'q2'},
        'q2': {'3': 'q2', '6': 'q1'}
    },
    initial_state='q0',
    final_states={'q2'}
)

dfa_2 = DFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'3', '6'},
    transitions={
        'q0': {'6': 'q0', '3': 'q1'},
        'q1': {'6': 'q0', '3': 'q2'},
        'q2': {'6': 'q2', '3': 'q1'}
    },
    initial_state='q0',
    final_states={'q1'}
)

dfa_3 = DFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'3', '6'},
    transitions={
        'q0': {'6': 'q0', '3': 'q1'},
        'q1': {'6': 'q0', '3': 'q2'},
        'q2': {'6': 'q2', '3': 'q1'}
    },
    initial_state='q0',
    final_states={'q2'}
)

dfa_4 = DFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'3', '6'},
    transitions={
        'q0': {'6': 'q0', '3': 'q1'},
        'q1': {'3': 'q2', '6': 'q0'},
        'q2': {'3': 'q1', '6': 'q2'}
    },
    initial_state='q0',
    final_states={'q1'}
)

dfa_5 = DFA(
    states={'q0', 'q1', 'q2', 'q3', 'q4'},
    input_symbols={'3', '6'},
    transitions={
        'q0': {'6': 'q0', '3': 'q1'},
        'q1': {'6': 'q2', '3': 'q1'},      # NEED TO UPDATE THIS in the drawing
        'q2': {'3': 'q1', '6': 'q3'},
        'q3': {'3': 'q1', '6': 'q0'},
        'q4': {'3': 'q4', '6': 'q4'}
    },
    initial_state='q0',
    final_states={'q2'}
)

dfa_6 = DFA(
    states={'q0', 'q1', 'q2', 'q3', 'q4'},
    input_symbols={'3', '6'},
    transitions={
        'q0': {'6': 'q0', '3': 'q1'},
        # NEED TO UPDATE THIS in the drawing
        'q1': {'6': 'q2', '3': 'q1'},
        'q2': {'3': 'q1', '6': 'q3'},
        'q3': {'3': 'q1', '6': 'q0'},
        'q4': {'3': 'q4', '6': 'q4'}
    },
    initial_state='q0',
    final_states={'q3'}
)


def generate_string(length, dfa, final_results):
    retval = ''
    current_state = dfa.initial_state

    # for i in range(length):
    #     transition = str(randrange(0, len(dfa.transitions[current_state])))
    #     current_state = dfa.transitions[current_state][transition]
    #     retval += transition

    retry = True
    while retry is True:
        retry = False
        while ((len(retval) < length) or (current_state not in dfa.final_states)) and (retval not in final_results):
            transition = str(randrange(0, len(dfa.transitions[current_state])))
            current_state = dfa.transitions[current_state][transition]
            retval += transition
            # print(retval)

            if (len(retval) > (length * 3)):
                print("ESCAPING")
                break

        try:
            dfa.validate_input(retval)
        except RejectionError as e:
            retry = True
            print("REJECTED: {}".format(e))

    return retval


if __name__ == "__main__":
    pass

    MIN_LEN = 5
    MAX_LEN = 1000
    NUM_SAMPLES_PER_LEN = 1000
    DATA_WIDTH = 10000
    VALID_OUTFILE = 'dfa_0.npy'
    INVALID_OUTFILE = 'dfa_1.npy'

    total_samples = (MAX_LEN - MIN_LEN + 1) * NUM_SAMPLES_PER_LEN
    print('total_samples: {}'.format(total_samples))

    valid_data = np.zeros([total_samples, DATA_WIDTH, 1], dtype=np.uint8)
    invalid_data = np.zeros([total_samples, DATA_WIDTH, 1], dtype=np.uint8)
    # b = np.zeros(length - len(a), dtype=np.uint8)
    # c = np.append(a, b)

    i_valid = 0
    i_invalid = 0
    for length in range(MIN_LEN, MAX_LEN + 1):
        print("Creating samples strings of length {}".format(length))
        ctr_valid = 0
        ctr_invalid = 0
        valid_starting_pos = i_valid
        invalid_starting_pos = i_invalid
        POSSIBLE_SAMPLES = 2**length
        # print('\tFor strings of length {}, {} samples can be produced.'
        #       '\n\t{} samples were requested.'
        #       '\n\tGenerating {} samples'.format(length, POSSIBLE_SAMPLES, NUM_SAMPLES_PER_LEN, min(NUM_SAMPLES_PER_LEN, POSSIBLE_SAMPLES)))
        attempt = 1
        while (ctr_valid < min(NUM_SAMPLES_PER_LEN, POSSIBLE_SAMPLES)) and (ctr_invalid < min(NUM_SAMPLES_PER_LEN, POSSIBLE_SAMPLES)):
            a = np.random.randint(
                low=1, high=3, size=length, dtype=np.uint8) * 3
            z = np.zeros(DATA_WIDTH - len(a), dtype=np.uint8)
            s = np.vstack(np.append(a, z))

            try:

                dfa_0.validate_input(a)

                # a = np.vstack(np.append(a, z))
                if (ctr_valid < min(NUM_SAMPLES_PER_LEN, POSSIBLE_SAMPLES)):
                    found = False
                    for j in range(valid_starting_pos, valid_starting_pos+ctr_valid):
                        if np.array_equal(valid_data[j], s):
                            found = True
                            break

                    if not found:
                        valid_data[i_valid] = s
                        i_valid += 1
                        ctr_valid += 1
                    else:
                        attempt += 1

            except RejectionError as e:
                # print("Rejection ERROR!: {}".format(e))

                if (ctr_invalid < min(NUM_SAMPLES_PER_LEN, POSSIBLE_SAMPLES)):
                    found = False
                    for j in range(invalid_starting_pos, invalid_starting_pos+ctr_invalid):
                        if np.array_equal(invalid_data[j], s):
                            found = True
                            break

                    if not found:
                        invalid_data[i_invalid] = s
                        i_invalid += 1
                        ctr_invalid += 1
                    else:
                        attempt += 1

            if attempt % 50 == 0:
                print('.', end='')
            if attempt % 500 == 0:
                print('')
            if attempt % 1000 == 0:
                break

        # print('')
        print('Created {} valid and {} invalid strings of length {}'.format(
            ctr_valid, ctr_invalid, length))
        # time.sleep(5)
    np.save(VALID_OUTFILE, valid_data[:ctr_valid])
    np.save(INVALID_OUTFILE, invalid_data[:ctr_invalid])

    # final_results = []
    # dfas = [dfa_0, dfa_1, dfa_2, dfa_3, dfa_4, dfa_5, dfa_6]
    # num_trials = 1000
    # minimum_string_length = 1000

    # for i in range(len(dfas)):
    #     interim_results = []
    #     while len(interim_results) < num_trials:
    #         s = generate_string(minimum_string_length, dfas[i], final_results)

    #         if s not in interim_results:
    #             # print(s, interim_results)
    #             if s not in final_results:
    #                 # print(s, final_results)
    #                 interim_results.append(s)
    #                 print('{}: {}'.format(i, len(interim_results)))

    #     # print(len(interim_results))
    #     for ir in interim_results:
    #         final_results.append(('dfa_{}'.format(i), ir))

    # # print(final_results)

    # # Check to make sure that only one match exists
    # for i in range(len(final_results)):
    #     match_count = 0
    #     try:
    #         for dfa in dfas:
    #             dfa.validate_input(final_results[i][1])
    #             match_count += 1

    #     except RejectionError as e:
    #         # print("Rejection ERROR!: {}".format(e))
    #         pass

    #     if match_count > 1:
    #         print("error")

    # with open('trial_data.csv', 'w') as f:
    #     for fr in final_results:

    #         print('{},{}\n'.format(fr[0], fr[1]))
    #         f.write('{},{}\n'.format(fr[0], fr[1]))
