from common import COMMON


def do_nothing(*list_args):
    return list_args[0]


def get_functions():
    from pypgm.factor import dlinear
    
    functions = {
        'init_parameters': do_nothing,
        'multiply_by': dlinear.multiply_parameter_sets,
        'divide_by': do_nothing,
        'marginalize_to': do_nothing,
        'normalize': do_nothing,
        }

    if COMMON.error_check:
        functions.update({
            'pre_init_parameters_check': do_nothing,
            'post_init_parameters_check': do_nothing,
            'pre_multiply_by_check': do_nothing,
            'post_multiply_by_check': do_nothing,
            'pre_divide_by_check': do_nothing,
            'post_divide_by_check': do_nothing,
            'pre_marginalize_to_check': do_nothing,
            'post_marginalize_to_check': do_nothing,
            'pre_normalize_check': do_nothing,
            'post_normalize_check': do_nothing,
        })

    else:
        functions.update({
            'pre_init_parameters_check': do_nothing,
            'post_init_parameters_check': do_nothing,
            'pre_multiply_by_check': do_nothing,
            'post_multiply_by_check': do_nothing,
            'pre_divide_by_check': do_nothing,
            'post_divide_by_check': do_nothing,
            'pre_marginalize_to_check': do_nothing,
            'post_marginalize_to_check': do_nothing,
            'pre_normalize_check': do_nothing,
            'post_normalize_check': do_nothing,
        })

    return functions
