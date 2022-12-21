from factor import Factor

import scalar_normal
import vector_normal
import gamma
import dlinear
import dsigmoid

class FACTOR_TYPE(object):
    """Class containing all factor types"""
    SCALAR_NORMAL = 1
    VECTOR_NORMAL = 2
    GAMMA = 3
    DLINEAR = 4
    DSIGMOID = 5


FACTOR_FUNCTIONS = {
    FACTOR_TYPE.SCALAR_NORMAL : scalar_normal.get_functions(),
    FACTOR_TYPE.VECTOR_NORMAL : vector_normal.get_functions(),
    FACTOR_TYPE.GAMMA : gamma.get_functions(),
    FACTOR_TYPE.DLINEAR : dlinear.get_functions(),
    FACTOR_TYPE.DSIGMOID : dsigmoid.get_functions(),
}
