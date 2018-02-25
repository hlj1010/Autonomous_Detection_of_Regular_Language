from random import randrange

from automata.fa.dfa import DFA
from automata.shared.exceptions import RejectionError

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
        'q0': {'1': 'q0', '0': 'q1'},
        'q1': {'1': 'q2', '0': 'q1'},       # NEED TO UPDATE THIS in the drawing
        'q2': {'0': 'q1', '1': 'q3'},
        'q3': {'0': 'q1', '1': 'q0'},
        'q4': {'0': 'q4', '1': 'q4'}
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
    # s = r_generate_string(500, dfa.initial_state)

    final_results = []
    dfas = [dfa_0, dfa_1, dfa_2, dfa_3, dfa_4, dfa_5, dfa_6]
    num_trials = 1000
    minimum_string_length = 1000

    for i in range(len(dfas)):
        interim_results = []
        while len(interim_results) < num_trials:
            s = generate_string(minimum_string_length, dfas[i], final_results)

            if s not in interim_results:
                # print(s, interim_results)
                if s not in final_results:
                    # print(s, final_results)
                    interim_results.append(s)
                    print('{}: {}'.format(i, len(interim_results)))

        # print(len(interim_results))
        for ir in interim_results:
            final_results.append(('dfa_{}'.format(i), ir))

    # print(final_results)

    # Check to make sure that only one match exists
    for i in range(len(final_results)):
        match_count = 0
        try:
            for dfa in dfas:
                dfa.validate_input(final_results[i][1])
                match_count += 1

        except RejectionError as e:
            # print("Rejection ERROR!: {}".format(e))
            pass

        if match_count > 1:
            print("error")

    with open('trial_data.csv', 'w') as f:
        for fr in final_results:

            print('{},{}\n'.format(fr[0], fr[1]))
            f.write('{},{}\n'.format(fr[0], fr[1]))
