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
        'q1': {'1': 'q2', '0': 'q4'},
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
        'q1': {'1': 'q2', '0': 'q4'},
        'q2': {'0': 'q1', '1': 'q3'},
        'q3': {'0': 'q1', '1': 'q0'},
        'q4': {'0': 'q4', '1': 'q4'}
    },
    initial_state='q0',
    final_states={'q3'}
)


def generate_string(length, dfa):
    retval = ''
    current_state = dfa.initial_state

    for i in range(length):
        transition = str(randrange(0, len(dfa.transitions[current_state])))
        current_state = dfa.transitions[current_state][transition]
        retval += transition

    while current_state not in dfa.final_states:
        transition = str(randrange(0, len(dfa.transitions[current_state])))
        current_state = dfa.transitions[current_state][transition]
        retval += transition

    return retval


if __name__ == "__main__":
    # s = r_generate_string(500, dfa.initial_state)

    interim_results = []
    final_results = []
    dfas = [dfa_0, dfa_1, dfa_2, dfa_3, dfa_4, dfa_5, dfa_6]
    trials = 1000

    for i in range(len(dfas)):
        while len(interim_results) < trials:
            s = generate_string(trials, dfas[i])
            if s not in interim_results:
                interim_results.append(s)

        print(len(interim_results))
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
            pass

        if match_count > 1:
            print("error")

    with open('trial_data.csv', 'w') as f:
        for fr in final_results:
            f.write('{}, {}\n'.format(fr[0], fr[1]))
