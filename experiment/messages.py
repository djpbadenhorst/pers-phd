import ipdb

import os

from IO import *

from mathematics import *


def run_instruction(instruction):
    if instruction[0] == 'upd_belief':
        _update_belief(instruction[1:])
    elif instruction[0] == 'upd_message':
        _update_message(instruction[1:])
    else:
        raise Exception("ERROR")


def _update_belief(factor_name):
    factor_name = factor_name[0]
    factor_parameters = read_priors(factor_name)
    connections = read_connections(factor_name)
    for factor_name_from in connections:
        inc_message = read_messages(factor_name_from, factor_name)
        factor_parameters = multiply_message(factor_parameters, inc_message)

    store_beliefs(factor_name, factor_parameters)


def _update_message(factor_names):
    factor_from = factor_names[0]
    factor_to = factor_names[1]
    
    belief_from = read_beliefs(factor_from)
    belief_to = read_beliefs(factor_to)
    
    inc_message = read_messages(factor_to, factor_from)
    out_message = read_messages(factor_from, factor_to)
    
    out_message = approximate_message(belief_from, belief_to, inc_message, out_message)

    store_messages(factor_from, factor_to, out_message)

