from common import TEST

import unittest


class test_saa_basic_gaussian(unittest.TestCase):
    def tearDown(self):
        """Destroys necessary variables"""

        import shutil

        shutil.rmtree('./saa/')

        
    def test(self):
        """Basic example containing gaussian fnodes only"""

        TEST.LOG("START - TEST_SAA_BASIC_GAUSSIAN", 1)
        TEST.LOG(self.test.__doc__, 2)

        import numpy as np

        EVAL_VALUES = np.linspace(0,10,101)

        from copy import deepcopy
        
        from utils.sdict import SDict

        from pypgm.factor.vector_normal.pdf import calculate_pdf
        
        from pypgm import PGM
        
        from pypgm.factor import FACTOR_TYPE, Factor, vector_normal, scalar_normal

        pgm = PGM('./saa/')

        pgm.add_fnode('f(a,b)',
                      ftype=FACTOR_TYPE.VECTOR_NORMAL,
                      par_form=vector_normal.PARAMETER_FORM.STANDARD,
                      cov_form=vector_normal.COVARIANCE_FORM.FULL,
                      vars=['a', 'b'],
                      mean=[3, 9],
                      cov=[[1, 0.9], [0.9, 5]])
        pgm.add_fnode('f(b,c)',
                      ftype=FACTOR_TYPE.VECTOR_NORMAL,
                      par_form=vector_normal.PARAMETER_FORM.STANDARD,
                      cov_form=vector_normal.COVARIANCE_FORM.FULL,
                      vars=['b', 'c'],
                      mean=[3, 4],
                      cov=[[3, 0.6], [0.6, 1]])
        pgm.add_fnode('f(c,d)',
                      ftype=FACTOR_TYPE.VECTOR_NORMAL,
                      par_form=vector_normal.PARAMETER_FORM.STANDARD,
                      cov_form=vector_normal.COVARIANCE_FORM.FULL,
                      vars=['c', 'd'],
                      mean=[3, 5],
                      cov=[[1, -0.4], [-0.4, 2]])
        pgm.add_fnode('f(d,e)',
                      ftype=FACTOR_TYPE.VECTOR_NORMAL,
                      par_form=vector_normal.PARAMETER_FORM.STANDARD,
                      cov_form=vector_normal.COVARIANCE_FORM.FULL,
                      vars=['d', 'e'],
                      mean=[8, 6],
                      cov=[[3, -0.7], [-0.7, 4]])

        pgm.connect_fnodes('f(a,b)', 'f(b,c)', ['b'])
        pgm.connect_fnodes('f(b,c)', 'f(c,d)', ['c'])
        pgm.connect_fnodes('f(c,d)', 'f(d,e)', ['d'])

        pgm.update_belief('f(a,b)')
        pgm.update_message('f(a,b)', 'f(b,c)')
        pgm.update_belief('f(b,c)')
        pgm.update_message('f(b,c)', 'f(c,d)')
        pgm.update_belief('f(c,d)')
        pgm.update_message('f(c,d)', 'f(d,e)')
        pgm.update_belief('f(d,e)')
        pgm.update_message('f(d,e)', 'f(c,d)')
        pgm.update_belief('f(c,d)')
        pgm.update_message('f(c,d)', 'f(b,c)')
        pgm.update_belief('f(b,c)')
        pgm.update_message('f(b,c)', 'f(a,b)')
        pgm.update_belief('f(a,b)')

        f_a_b = Factor(**pgm.get_prior_parameters('f(a,b)'))
        f_b_c = Factor(**pgm.get_prior_parameters('f(b,c)'))
        f_c_d = Factor(**pgm.get_prior_parameters('f(c,d)'))
        f_d_e = Factor(**pgm.get_prior_parameters('f(d,e)'))

        f_prod = deepcopy(f_a_b)
        f_prod.multiply_by(f_b_c)
        f_prod.multiply_by(f_c_d)
        f_prod.multiply_by(f_d_e)

        f_a_b = deepcopy(f_prod)
        f_a_b.marginalize_to(['a', 'b'])
        f_b_c = deepcopy(f_prod)
        f_b_c.marginalize_to(['b', 'c'])
        f_c_d = deepcopy(f_prod)
        f_c_d.marginalize_to(['c', 'd'])
        f_d_e = deepcopy(f_prod)
        f_d_e.marginalize_to(['d', 'e'])

        f_a_b_exact = calculate_pdf(EVAL_VALUES, **f_a_b.parameters)
        f_a_b_est = calculate_pdf(EVAL_VALUES, **pgm.get_belief_parameters('f(a,b)'))
        f_b_c_exact = calculate_pdf(EVAL_VALUES, **f_b_c.parameters)
        f_b_c_est = calculate_pdf(EVAL_VALUES, **pgm.get_belief_parameters('f(b,c)'))
        f_c_d_exact = calculate_pdf(EVAL_VALUES, **f_c_d.parameters)
        f_c_d_est = calculate_pdf(EVAL_VALUES, **pgm.get_belief_parameters('f(c,d)'))        
        f_d_e_exact = calculate_pdf(EVAL_VALUES, **f_d_e.parameters)
        f_d_e_est = calculate_pdf(EVAL_VALUES, **pgm.get_belief_parameters('f(d,e)'))
        
        if True:
            import pylab as plt

            plt.figure(figsize=(15,8))
            plt.suptitle('Estimated (top) Exact (bottom)')
            plt.subplot(211)
            plt.imshow(f_a_b_est, origin='lower', aspect='auto', interpolation="bicubic")
            plt.xticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.yticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.xlabel('b')
            plt.ylabel('a')
            plt.subplot(212)
            plt.imshow(f_a_b_exact, origin='lower', aspect='auto', interpolation="bicubic")
            plt.xticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.yticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.xlabel('b')
            plt.ylabel('a')
            plt.savefig('./images/examples/saa_a')
            plt.close()

            plt.figure(figsize=(15,8))
            plt.suptitle('Estimated (top) Exact (bottom)')
            plt.subplot(211)
            plt.imshow(f_b_c_est, origin='lower', aspect='auto', interpolation="bicubic")
            plt.xticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.yticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.xlabel('c')
            plt.ylabel('b')
            plt.subplot(212)
            plt.imshow(f_b_c_exact, origin='lower', aspect='auto', interpolation="bicubic")
            plt.xticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.yticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.xlabel('c')
            plt.ylabel('b')
            plt.savefig('./images/examples/saa_b')
            plt.close()

            plt.figure(figsize=(15,8))
            plt.suptitle('Estimated (top) Exact (bottom)')
            plt.subplot(211)
            plt.imshow(f_c_d_est, origin='lower', aspect='auto', interpolation="bicubic")
            plt.xticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.yticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.xlabel('d')
            plt.ylabel('c')
            plt.subplot(212)
            plt.imshow(f_c_d_exact, origin='lower', aspect='auto', interpolation="bicubic")
            plt.xticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.yticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.xlabel('d')
            plt.ylabel('c')
            plt.savefig('./images/examples/saa_c')
            plt.close()

            plt.figure(figsize=(15,8))
            plt.suptitle('Estimated (top) Exact (bottom)')
            plt.subplot(211)
            plt.imshow(f_d_e_est, origin='lower', aspect='auto', interpolation="bicubic")
            plt.xticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.yticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.xlabel('e')
            plt.ylabel('d')
            plt.subplot(212)
            plt.imshow(f_d_e_exact, origin='lower', aspect='auto', interpolation="bicubic")
            plt.xticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.yticks(np.arange(len(EVAL_VALUES))[::10], EVAL_VALUES[::10])
            plt.xlabel('e')
            plt.ylabel('d')
            plt.savefig('./images/examples/saa_d')
            plt.close()

        TEST.LOG("EXAMPLE COMPLETE", 1)

