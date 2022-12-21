from message import Message

from pypgm.factor import FACTOR_TYPE

from pypgm.message import scalar_normal_to_scalar_normal
from pypgm.message import vector_normal_to_vector_normal
from pypgm.message import scalar_normal_to_vector_normal
from pypgm.message import vector_normal_to_scalar_normal
from pypgm.message import vector_normal_to_dlinear
from pypgm.message import dlinear_to_vector_normal
from pypgm.message import scalar_normal_to_dlinear
from pypgm.message import dlinear_to_scalar_normal
from pypgm.message import gamma_to_dlinear
from pypgm.message import dlinear_to_gamma
from pypgm.message import scalar_normal_to_dsigmoid
from pypgm.message import dsigmoid_to_scalar_normal


class MESSAGE_TYPE(object):
    SCALAR_NORMAL_TO_SCALAR_NORMAL = (FACTOR_TYPE.SCALAR_NORMAL << 8) + FACTOR_TYPE.SCALAR_NORMAL
    VECTOR_NORMAL_TO_VECTOR_NORMAL = (FACTOR_TYPE.VECTOR_NORMAL << 8) + FACTOR_TYPE.VECTOR_NORMAL
    SCALAR_NORMAL_TO_VECTOR_NORMAL = (FACTOR_TYPE.SCALAR_NORMAL << 8) + FACTOR_TYPE.VECTOR_NORMAL
    VECTOR_NORMAL_TO_SCALAR_NORMAL = (FACTOR_TYPE.VECTOR_NORMAL << 8) + FACTOR_TYPE.SCALAR_NORMAL
    VECTOR_NORMAL_TO_DLINEAR = (FACTOR_TYPE.VECTOR_NORMAL << 8) + FACTOR_TYPE.DLINEAR
    DLINEAR_TO_VECTOR_NORMAL = (FACTOR_TYPE.DLINEAR << 8) + FACTOR_TYPE.VECTOR_NORMAL
    SCALAR_NORMAL_TO_DLINEAR = (FACTOR_TYPE.SCALAR_NORMAL << 8) + FACTOR_TYPE.DLINEAR
    DLINEAR_TO_SCALAR_NORMAL = (FACTOR_TYPE.DLINEAR << 8) + FACTOR_TYPE.SCALAR_NORMAL
    GAMMA_TO_DLINEAR = (FACTOR_TYPE.GAMMA << 8) + FACTOR_TYPE.DLINEAR
    DLINEAR_TO_GAMMA = (FACTOR_TYPE.DLINEAR << 8) + FACTOR_TYPE.GAMMA
    SCALAR_NORMAL_TO_DSIGMOID = (FACTOR_TYPE.SCALAR_NORMAL << 8) + FACTOR_TYPE.DSIGMOID
    DSIGMOID_TO_SCALAR_NORMAL = (FACTOR_TYPE.DSIGMOID << 8) + FACTOR_TYPE.SCALAR_NORMAL


FUNCTIONS = {
    MESSAGE_TYPE.SCALAR_NORMAL_TO_SCALAR_NORMAL: scalar_normal_to_scalar_normal.get_functions(),
    MESSAGE_TYPE.VECTOR_NORMAL_TO_VECTOR_NORMAL: vector_normal_to_vector_normal.get_functions(),
    MESSAGE_TYPE.SCALAR_NORMAL_TO_VECTOR_NORMAL: scalar_normal_to_vector_normal.get_functions(),
    MESSAGE_TYPE.VECTOR_NORMAL_TO_SCALAR_NORMAL: vector_normal_to_scalar_normal.get_functions(),
    MESSAGE_TYPE.VECTOR_NORMAL_TO_DLINEAR: vector_normal_to_dlinear.get_functions(),
    MESSAGE_TYPE.DLINEAR_TO_VECTOR_NORMAL: dlinear_to_vector_normal.get_functions(),
    MESSAGE_TYPE.SCALAR_NORMAL_TO_DLINEAR: scalar_normal_to_dlinear.get_functions(),
    MESSAGE_TYPE.DLINEAR_TO_SCALAR_NORMAL: dlinear_to_scalar_normal.get_functions(),
    MESSAGE_TYPE.GAMMA_TO_DLINEAR: gamma_to_dlinear.get_functions(),
    MESSAGE_TYPE.DLINEAR_TO_GAMMA: dlinear_to_gamma.get_functions(),
    MESSAGE_TYPE.SCALAR_NORMAL_TO_DSIGMOID: scalar_normal_to_dsigmoid.get_functions(),
    MESSAGE_TYPE.DSIGMOID_TO_SCALAR_NORMAL: dsigmoid_to_scalar_normal.get_functions(),
}
