from common import COMMON


def do_nothing(*list_args):
    return list_args[0]


def get_functions():
    from pypgm.message import scalar_normal_to_dlinear

    functions = {
        'init_message': scalar_normal_to_dlinear.init_message,
        'est_message': scalar_normal_to_dlinear.est_message,
        }

    if COMMON.error_check:
        functions.update({
            'pre_init_message_check': do_nothing,
            'post_init_message_check': do_nothing,
            'pre_est_message_check': do_nothing,
            'post_est_message_check': do_nothing,
        })

    else:
        functions.update({
            'pre_init_message_check': do_nothing,
            'post_init_message_check': do_nothing,
            'pre_est_message_check': do_nothing,
            'post_est_message_check': do_nothing,
        })

    return functions
