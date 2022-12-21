from common import TEST

import unittest


class test_sae_basic_single_dlinearg(unittest.TestCase):
    def tearDown(self):
        """Destroys necessary variables"""

        import shutil

        shutil.rmtree('./sae/')
        
        
    def test(self):
        """Example for basic linear regression"""

        TEST.LOG("START - EXAMPLE", 1)
        TEST.LOG(self.test.__doc__, 2)
        
        import numpy as np
        np.random.seed(0)

        NUMBER_SAMPLES = 10
        NUM_ITERATIONS = 10

        W1_TRUE = [[-1., 1.],
                   [ 2., 0.],
                   [ 0., 1.]]
        W2_TRUE = [[ 1.],
                   [-1.]]

        INP_DIM = np.shape(W1_TRUE)[0]
        LAT_DIM = np.shape(W1_TRUE)[1]

        INP_MATRIX = np.random.normal(0, 5, (NUMBER_SAMPLES, INP_DIM))
        LAT_VALS = INP_MATRIX.dot(W1_TRUE)
        OUT_VALS = LAT_VALS.dot(W2_TRUE)
        OUT_NOISE = 0.000001
        OUT_VALS = OUT_VALS + np.random.normal(0, OUT_NOISE, (NUMBER_SAMPLES,1))

        INP_COV = np.diag(np.ones(INP_DIM))*0.0001
        OUT_COV = 0.0001
        
        LAT_COV = 10000
        
        W1_COV = np.diag([1e3]*INP_DIM).tolist()
        W2_COV = np.diag([1e3]*LAT_DIM).tolist()
        
        P1_ALPHA, P1_BETA = [1+1e-3, 1e-10]
        P2_ALPHA, P2_BETA = [1+1e-3, 1e-10]

        W1_EVAL_VALUES = np.arange(-20,20,0.5)
        W2_EVAL_VALUES = np.arange(-5,5,0.5)
        P1_EVAL_VALUES = np.linspace(1e-10,1e-1,100)
        P2_EVAL_VALUES = np.linspace(1e-10,1e-1,100)

        import scipy as sp
        
        from utils.sdict import SDict

        from dist.gamma import calculate_gamma_pdf
        from dist.gaussian import calculate_gaussian_pdf

        from pypgm import PGM
        from pypgm.factor import FTYPE, Factor

        from pypgm.factor.gaussian import to_standard
        from pypgm.factor.dlinearg import convert_canonicalset_to_standardset

        pgm = PGM('./sae/')
        images_folder = './../images/examples/'

        def create_nodes():
            def create_weights_nodes():
                for lat_id in range(LAT_DIM):
                    pgm.add_fnode('f(w1_{})'.format(str(lat_id).zfill(2)),
                                  ftype=FTYPE.GAUSSIAN,
                                  form='standard',
                                  vars=['w1_{}_{}'.format(str(lat_id).zfill(2), str(i).zfill(2)) for i in range(INP_DIM)],
                                  mean=np.random.rand(INP_DIM),
                                  cov=W1_COV)
                    pgm.add_fnode('f(p1_{})'.format(str(lat_id).zfill(2)),
                                  ftype=FTYPE.GAMMA,
                                  vars=['p1_{}'.format(str(lat_id).zfill(2))],
                                  alpha=P1_ALPHA,
                                  beta=P1_BETA)
                    
                pgm.add_fnode('f(w2)',
                              ftype=FTYPE.GAUSSIAN,
                              form='standard',
                              vars=['w2_{}'.format(str(i).zfill(2)) for i in range(LAT_DIM)],
                              mean=np.random.rand(LAT_DIM),
                              cov=W2_COV)
                pgm.add_fnode('f(p2)',
                              ftype=FTYPE.GAMMA,
                              vars=['p2'],
                              alpha=P2_ALPHA,
                              beta=P2_BETA)

            create_weights_nodes()

            def create_first_layer_nodes():
                for dat_id, x in enumerate(INP_MATRIX):
                    # Create nodes for inputs
                    pgm.add_fnode('f(x_{})'.format(str(dat_id).zfill(2)),
                                  ftype=FTYPE.GAUSSIAN,
                                  form='standard',
                                  vars=['x_{}_{}'.format(str(dat_id).zfill(2), str(i).zfill(2)) for i in range(INP_DIM)],
                                  mean=x.tolist(),
                                  cov=INP_COV)

                    # Create nodes for first latent variable
                    for lat_id in range(LAT_DIM):
                        pgm.add_fnode('f(z_{}_{})'.format(str(dat_id).zfill(2), str(lat_id).zfill(2)),
                                      ftype=FTYPE.GAUSSIAN,
                                      form='standard',
                                      vars=['z_{}_{}'.format(str(dat_id).zfill(2), str(lat_id).zfill(2))],
                                      mean=[0],
                                      cov=[[LAT_COV]])
                        pgm.add_fnode('f(x_{},z_{}_{},w1_{},p1_{})'.format(str(dat_id).zfill(2),
                                                                            str(dat_id).zfill(2),
                                                                            str(lat_id).zfill(2),
                                                                            str(lat_id).zfill(2),
                                                                            str(lat_id).zfill(2)),
                                      ftype=FTYPE.DLINEARG,
                                      x_vars=['x_{}_{}'.format(str(dat_id).zfill(2), str(i).zfill(2)) for i in range(INP_DIM)],
                                      y_vars=['z_{}_{}'.format(str(dat_id).zfill(2), str(lat_id).zfill(2))],
                                      w_vars=['w1_{}_{}'.format(str(lat_id).zfill(2), str(i).zfill(2)) for i in range(INP_DIM)],
                                      p_vars=['p1_{}'.format(str(lat_id).zfill(2))])

            create_first_layer_nodes()

            def create_output_nodes():
                for dat_id, y in enumerate(OUT_VALS):
                    # Create nodes for second latent variable vector
                    pgm.add_fnode('f(z_{})'.format(str(dat_id).zfill(2)),
                                  ftype=FTYPE.GAUSSIAN,
                                  form='standard',
                                  vars=['z_{}_{}'.format(str(dat_id).zfill(2), str(i).zfill(2)) for i in range(LAT_DIM)],
                                  mean=np.zeros(LAT_DIM).tolist(),
                                  cov=np.diag([LAT_COV]*LAT_DIM).tolist())
                    
                    # Create nodes for output
                    pgm.add_fnode('f(y_{})'.format(str(dat_id).zfill(2)),
                                  ftype=FTYPE.GAUSSIAN,
                                  form='standard',
                                  vars=['y_{}'.format(str(dat_id).zfill(2))],
                                  mean=[y],
                                  cov=[[OUT_COV]])
                    pgm.add_fnode('f(z_{},y_{},w2,p2)'.format(str(dat_id).zfill(2), str(dat_id).zfill(2)),
                                  ftype=FTYPE.DLINEARG,
                                  x_vars=['z_{}_{}'.format(str(dat_id).zfill(2), str(i).zfill(2)) for i in range(LAT_DIM)],
                                  y_vars=['y_{}'.format(str(dat_id).zfill(2))],
                                  w_vars=['w2_{}'.format(str(i).zfill(2)) for i in range(LAT_DIM)],
                                  p_vars=['p2'])

            create_output_nodes()

        create_nodes()
        
        def connect_nodes():
            def connect_first_layer_nodes():
                # Connect nodes in first latent node layer
                for dat_id in range(NUMBER_SAMPLES):
                    for lat_id in range(LAT_DIM):
                        factor_name = 'f(x_{},z_{}_{},w1_{},p1_{})'.format(str(dat_id).zfill(2), str(dat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2))
                        
                        pgm.connect_fnodes('f(x_{})'.format(str(dat_id).zfill(2)), factor_name,
                                           ['x_{}_{}'.format(str(dat_id).zfill(2), str(i).zfill(2)) for i in range(INP_DIM)])

                        pgm.connect_fnodes('f(z_{}_{})'.format(str(dat_id).zfill(2), str(lat_id).zfill(2)), factor_name,
                                           ['z_{}_{}'.format(str(dat_id).zfill(2), str(lat_id).zfill(2))])

                        pgm.connect_fnodes('f(w1_{})'.format(str(lat_id).zfill(2)), factor_name,
                                           ['w1_{}_{}'.format(str(lat_id).zfill(2), str(i).zfill(2)) for i in range(INP_DIM)])

                        pgm.connect_fnodes('f(p1_{})'.format(str(lat_id).zfill(2)), factor_name,
                                           ['p1_{}'.format(str(lat_id).zfill(2))])

                        pgm.connect_fnodes('f(z_{}_{})'.format(str(dat_id).zfill(2), str(lat_id).zfill(2)), 'f(z_{})'.format(str(dat_id).zfill(2)),
                                           ['z_{}_{}'.format(str(dat_id).zfill(2), str(lat_id).zfill(2))])
         
            connect_first_layer_nodes()
            
            def connect_output_layer_nodes():
                # Connect nodes in output latent node layer
                for dat_id in range(NUMBER_SAMPLES):
                    factor_name = 'f(z_{},y_{},w2,p2)'.format(str(dat_id).zfill(2), str(dat_id).zfill(2))
                    
                    pgm.connect_fnodes('f(z_{})'.format(str(dat_id).zfill(2)), factor_name,
                                       ['z_{}_{}'.format(str(dat_id).zfill(2), str(i).zfill(2)) for i in range(LAT_DIM)])
                    
                    pgm.connect_fnodes('f(y_{})'.format(str(dat_id).zfill(2)), factor_name,
                                       ['y_{}'.format(str(dat_id).zfill(2))])
                    
                    pgm.connect_fnodes('f(w2)', factor_name,
                                       ['w2_{}'.format(str(i).zfill(2)) for i in range(LAT_DIM)])
                    
                    pgm.connect_fnodes('f(p2)', factor_name,
                                       ['p2'])

            connect_output_layer_nodes()
            
        connect_nodes()

        def message_passing():
            def mp_first_layer_delta_node(lat_id):
                for dat_id in range(NUMBER_SAMPLES):
                    factor_name = 'f(x_{},z_{}_{},w1_{},p1_{})'.format(str(dat_id).zfill(2), str(dat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2))
                    pgm.update_belief(factor_name)
                        
            def mp_first_layer_modval(lat_id):
                for dat_id in range(NUMBER_SAMPLES):
                    factor_name = 'f(x_{},z_{}_{},w1_{},p1_{})'.format(str(dat_id).zfill(2), str(dat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2))
                    pgm.update_message(factor_name, 'f(p1_{})'.format(str(lat_id).zfill(2)))
                pgm.update_belief('f(p1_{})'.format(str(lat_id).zfill(2)))
                print SDict(**pgm.get_belief_parameters('f(p1_{})'.format(str(lat_id).zfill(2))))
                for dat_id in range(NUMBER_SAMPLES):
                    factor_name = 'f(x_{},z_{}_{},w1_{},p1_{})'.format(str(dat_id).zfill(2), str(dat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2))
                    pgm.update_message('f(p1_{})'.format(str(lat_id).zfill(2)), factor_name)

            def mp_first_layer_weights(lat_id):
                for dat_id in range(NUMBER_SAMPLES):
                    factor_name = 'f(x_{},z_{}_{},w1_{},p1_{})'.format(str(dat_id).zfill(2), str(dat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2))
                    pgm.update_message(factor_name, 'f(w1_{})'.format(str(lat_id).zfill(2)))
                pgm.update_belief('f(w1_{})'.format(str(lat_id).zfill(2)))
                print SDict(**to_standard(pgm.get_belief_parameters('f(w1_{})'.format(str(lat_id).zfill(2)))))
                for dat_id in range(NUMBER_SAMPLES):
                    factor_name = 'f(x_{},z_{}_{},w1_{},p1_{})'.format(str(dat_id).zfill(2), str(dat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2))
                    pgm.update_message('f(w1_{})'.format(str(lat_id).zfill(2)), factor_name)

            def mp_first_layer_to_latent_vec(lat_id):
                for dat_id in range(NUMBER_SAMPLES):
                    factor_name = 'f(x_{},z_{}_{},w1_{},p1_{})'.format(str(dat_id).zfill(2), str(dat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2))
                    pgm.update_message(factor_name, 'f(z_{}_{})'.format(str(dat_id).zfill(2), str(lat_id).zfill(2)))
                    pgm.update_belief('f(z_{}_{})'.format(str(dat_id).zfill(2), str(lat_id).zfill(2)))
                    pgm.update_message('f(z_{}_{})'.format(str(dat_id).zfill(2), str(lat_id).zfill(2)),
                                       'f(z_{})'.format(str(dat_id).zfill(2)))

            def mp_latent_vec_to_output_layer():
                for dat_id in range(NUMBER_SAMPLES):
                    factor_name = 'f(z_{},y_{},w2,p2)'.format(str(dat_id).zfill(2), str(dat_id).zfill(2))
                    pgm.update_belief('f(z_{})'.format(str(dat_id).zfill(2)))
                    pgm.update_message('f(z_{})'.format(str(dat_id).zfill(2)),factor_name)

            def mp_output_layer_delta_node():
                for dat_id in range(NUMBER_SAMPLES):
                    factor_name = 'f(z_{},y_{},w2,p2)'.format(str(dat_id).zfill(2), str(dat_id).zfill(2))
                    pgm.update_belief(factor_name)

            def mp_output_layer_modval():
                for dat_id in range(NUMBER_SAMPLES):
                    factor_name = 'f(z_{},y_{},w2,p2)'.format(str(dat_id).zfill(2), str(dat_id).zfill(2))
                    pgm.update_message(factor_name, 'f(p2)')
                pgm.update_belief('f(p2)')
                print SDict(**pgm.get_belief_parameters('f(p2)'))
                for dat_id in range(NUMBER_SAMPLES):
                    factor_name = 'f(z_{},y_{},w2,p2)'.format(str(dat_id).zfill(2), str(dat_id).zfill(2))
                    pgm.update_message('f(p2)', factor_name)

            def mp_output_layer_weights():
                for dat_id in range(NUMBER_SAMPLES):
                    factor_name = 'f(z_{},y_{},w2,p2)'.format(str(dat_id).zfill(2), str(dat_id).zfill(2))
                    pgm.update_message(factor_name, 'f(w2)')
                pgm.update_belief('f(w2)')
                print SDict(**to_standard(pgm.get_belief_parameters('f(w2)')))
                for dat_id in range(NUMBER_SAMPLES):
                    factor_name = 'f(z_{},y_{},w2,p2)'.format(str(dat_id).zfill(2), str(dat_id).zfill(2))
                    pgm.update_message('f(w2)', factor_name)

            def mp_output_layer_to_latent_vec():
                for dat_id in range(NUMBER_SAMPLES):
                    factor_name = 'f(z_{},y_{},w2,p2)'.format(str(dat_id).zfill(2), str(dat_id).zfill(2))
                    pgm.update_message(factor_name, 'f(z_{})'.format(str(dat_id).zfill(2)))
                    pgm.update_belief('f(z_{})'.format(str(dat_id).zfill(2)))

            def mp_latent_vec_to_first_layer(lat_id):
                for dat_id in range(NUMBER_SAMPLES):
                    factor_name = 'f(x_{},z_{}_{},w1_{},p1_{})'.format(str(dat_id).zfill(2), str(dat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2), str(lat_id).zfill(2))
                    pgm.update_belief('f(z_{})'.format(str(dat_id).zfill(2)))
                    pgm.update_message('f(z_{})'.format(str(dat_id).zfill(2)), 'f(z_{}_{})'.format(str(dat_id).zfill(2), str(lat_id).zfill(2)))
                    pgm.update_belief('f(z_{}_{})'.format(str(dat_id).zfill(2), str(lat_id).zfill(2)))
                    pgm.update_message('f(z_{}_{})'.format(str(dat_id).zfill(2), str(lat_id).zfill(2)), factor_name)


            # Plot prior distributions
            if True:
                import pylab as plt

                image = calculate_gaussian_pdf(W1_EVAL_VALUES, **pgm.get_belief_parameters('f(w1_00)'))

                plt.figure(figsize=(15,8))
                plt.imshow(np.mean(image,2), origin='lower', aspect='auto', interpolation="bicubic")
                plt.xticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                plt.yticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                plt.xlabel('w1_00_1')
                plt.ylabel('w1_00_2')                
                plt.savefig(images_folder + 'sae_w1_00_12_0')
                plt.close()

                plt.figure(figsize=(15,8))
                plt.imshow(np.mean(image,1), origin='lower', aspect='auto', interpolation="bicubic")
                plt.xticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                plt.yticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                plt.xlabel('w1_00_1')
                plt.ylabel('w1_00_3')                
                plt.savefig(images_folder + 'sae_w1_00_23_0')
                plt.close()

                plt.figure(figsize=(15,8))
                plt.imshow(np.mean(image,0), origin='lower', aspect='auto', interpolation="bicubic")
                plt.xticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                plt.yticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                plt.xlabel('w1_00_2')
                plt.ylabel('w1_00_3')                
                plt.savefig(images_folder + 'sae_w1_00_13_0')

                image = calculate_gaussian_pdf(W1_EVAL_VALUES, **pgm.get_belief_parameters('f(w1_01)'))

                plt.figure(figsize=(15,8))
                plt.imshow(np.mean(image,2), origin='lower', aspect='auto', interpolation="bicubic")
                plt.xticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                plt.yticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                plt.xlabel('w1_01_1')
                plt.ylabel('w1_01_2')                
                plt.savefig(images_folder + 'sae_w1_01_12_0')
                plt.close()

                plt.figure(figsize=(15,8))
                plt.imshow(np.mean(image,1), origin='lower', aspect='auto', interpolation="bicubic")
                plt.xticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                plt.yticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                plt.xlabel('w1_01_1')
                plt.ylabel('w1_01_3')                
                plt.savefig(images_folder + 'sae_w1_01_23_0')
                plt.close()

                plt.figure(figsize=(15,8))
                plt.imshow(np.mean(image,0), origin='lower', aspect='auto', interpolation="bicubic")
                plt.xticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                plt.yticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                plt.xlabel('w1_01_2')
                plt.ylabel('w1_01_3')                
                plt.savefig(images_folder + 'sae_w1_01_13_0')
                plt.close()

                line = calculate_gamma_pdf(P1_EVAL_VALUES, **pgm.get_belief_parameters('f(p1_00)'))

                plt.figure(figsize=(15,8))
                plt.plot(line)
                plt.xticks(np.arange(len(P1_EVAL_VALUES))[::30], np.round(P1_EVAL_VALUES[::30],1))
                plt.ylim((0,1.05))
                plt.ylabel('p')
                plt.savefig(images_folder + 'sae_p1_00_0')
                plt.close()

                line = calculate_gamma_pdf(P1_EVAL_VALUES, **pgm.get_belief_parameters('f(p1_01)'))

                plt.figure(figsize=(15,8))
                plt.plot(line)
                plt.xticks(np.arange(len(P1_EVAL_VALUES))[::30], np.round(P1_EVAL_VALUES[::30],1))
                plt.ylim((0,1.05))
                plt.ylabel('p')
                plt.savefig(images_folder + 'sae_p1_01_0')
                plt.close()

                image = calculate_gaussian_pdf(W1_EVAL_VALUES, **pgm.get_belief_parameters('f(w2)'))

                plt.figure(figsize=(15,8))
                plt.imshow(image, origin='lower', aspect='auto', interpolation="bicubic")
                plt.xticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                plt.yticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                plt.xlabel('w2_1')
                plt.ylabel('w2_2')                
                plt.savefig(images_folder + 'sae_w2_0')
                plt.close()

                line = calculate_gamma_pdf(P1_EVAL_VALUES, **pgm.get_belief_parameters('f(p2)'))

                plt.figure(figsize=(15,8))
                plt.plot(line)
                plt.xticks(np.arange(len(P1_EVAL_VALUES))[::30], np.round(P1_EVAL_VALUES[::30],1))
                plt.ylim((0,1.05))
                plt.ylabel('p')
                plt.savefig(images_folder + 'sae_p2_0')
                plt.close()
                
            err_arr = []
            # Message passing
            for counter in range(NUM_ITERATIONS):
                print "\nITER {}".format(counter)

                mp_output_layer_delta_node()
                
                print "Second Layer Weights"
                mp_output_layer_weights()
                mp_output_layer_delta_node()
                
                print "Second Layer Precision"
                mp_output_layer_modval()
                mp_output_layer_delta_node()
                mp_output_layer_to_latent_vec()
                
                mp_latent_vec_to_first_layer(0)
                mp_latent_vec_to_first_layer(1)

                mp_first_layer_delta_node(0)
                mp_first_layer_delta_node(1)

                print "First Layer Weights - 1"
                mp_first_layer_weights(0)
                print "First Layer Weights - 2"
                mp_first_layer_weights(1)

                mp_first_layer_delta_node(0)
                mp_first_layer_delta_node(1)

                print "First Layer Precision - 1"
                mp_first_layer_modval(0)
                print "First Layer Precision - 2"
                mp_first_layer_modval(1)

                mp_first_layer_delta_node(0)
                mp_first_layer_delta_node(1)

                mp_first_layer_to_latent_vec(0)
                mp_first_layer_to_latent_vec(1)

                mp_latent_vec_to_output_layer()

                w1 = np.vstack([to_standard(pgm.get_belief_parameters('f(w1_{})'.format(str(lat_id).zfill(2))))['mean'] for lat_id in range(LAT_DIM)]).transpose()
                lat = INP_MATRIX.dot(w1)
                w2 = np.vstack(to_standard(pgm.get_belief_parameters('f(w2)'))['mean'])
                out = lat.dot(w2)

                err_arr.append(np.mean(np.abs(out - OUT_VALS)))
                print "Error Array"
                print err_arr

                # Plot current beliefs
                if True:
                    import pylab as plt
                    
                    image = calculate_gaussian_pdf(W1_EVAL_VALUES, **pgm.get_belief_parameters('f(w1_00)'))

                    plt.figure(figsize=(15,8))
                    plt.imshow(np.mean(image,2), origin='lower', aspect='auto', interpolation="bicubic")
                    plt.xticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                    plt.yticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                    plt.xlabel('w1_00_1')
                    plt.ylabel('w1_00_2')                
                    plt.savefig(images_folder + 'sae_w1_00_12_{}'.format(counter+1))
                    plt.close()

                    plt.figure(figsize=(15,8))
                    plt.imshow(np.mean(image,1), origin='lower', aspect='auto', interpolation="bicubic")
                    plt.xticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                    plt.yticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                    plt.xlabel('w1_00_1')
                    plt.ylabel('w1_00_3')                
                    plt.savefig(images_folder + 'sae_w1_00_23_{}'.format(counter+1))
                    plt.close()

                    plt.figure(figsize=(15,8))
                    plt.imshow(np.mean(image,0), origin='lower', aspect='auto', interpolation="bicubic")
                    plt.xticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                    plt.yticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                    plt.xlabel('w1_00_2')
                    plt.ylabel('w1_00_3')                
                    plt.savefig(images_folder + 'sae_w1_00_13_{}'.format(counter+1))

                    image = calculate_gaussian_pdf(W1_EVAL_VALUES, **pgm.get_belief_parameters('f(w1_01)'))

                    plt.figure(figsize=(15,8))
                    plt.imshow(np.mean(image,2), origin='lower', aspect='auto', interpolation="bicubic")
                    plt.xticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                    plt.yticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                    plt.xlabel('w1_01_1')
                    plt.ylabel('w1_01_2')                
                    plt.savefig(images_folder + 'sae_w1_01_12_{}'.format(counter+1))
                    plt.close()

                    plt.figure(figsize=(15,8))
                    plt.imshow(np.mean(image,1), origin='lower', aspect='auto', interpolation="bicubic")
                    plt.xticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                    plt.yticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                    plt.xlabel('w1_01_1')
                    plt.ylabel('w1_01_3')                
                    plt.savefig(images_folder + 'sae_w1_01_23_{}'.format(counter+1))
                    plt.close()

                    plt.figure(figsize=(15,8))
                    plt.imshow(np.mean(image,0), origin='lower', aspect='auto', interpolation="bicubic")
                    plt.xticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                    plt.yticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                    plt.xlabel('w1_01_2')
                    plt.ylabel('w1_01_3')                
                    plt.savefig(images_folder + 'sae_w1_01_13_{}'.format(counter+1))
                    plt.close()

                    line = calculate_gamma_pdf(P1_EVAL_VALUES, **pgm.get_belief_parameters('f(p1_00)'))

                    plt.figure(figsize=(15,8))
                    plt.plot(line)
                    plt.xticks(np.arange(len(P1_EVAL_VALUES))[::30], np.round(P1_EVAL_VALUES[::30],1))
                    plt.ylim((0,1.05))
                    plt.ylabel('p')
                    plt.savefig(images_folder + 'sae_p1_00_{}'.format(counter+1))
                    plt.close()

                    line = calculate_gamma_pdf(P1_EVAL_VALUES, **pgm.get_belief_parameters('f(p1_01)'))

                    plt.figure(figsize=(15,8))
                    plt.plot(line)
                    plt.xticks(np.arange(len(P1_EVAL_VALUES))[::30], np.round(P1_EVAL_VALUES[::30],1))
                    plt.ylim((0,1.05))
                    plt.ylabel('p')
                    plt.savefig(images_folder + 'sae_p1_01_{}'.format(counter+1))
                    plt.close()

                    image = calculate_gaussian_pdf(W1_EVAL_VALUES, **pgm.get_belief_parameters('f(w2)'))

                    plt.figure(figsize=(15,8))
                    plt.imshow(image, origin='lower', aspect='auto', interpolation="bicubic")
                    plt.xticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                    plt.yticks(np.arange(len(W1_EVAL_VALUES))[::10], W1_EVAL_VALUES[::10])
                    plt.xlabel('w2_1')
                    plt.ylabel('w2_2')                
                    plt.savefig(images_folder + 'sae_w2_{}'.format(counter+1))
                    plt.close()

                    line = calculate_gamma_pdf(P1_EVAL_VALUES, **pgm.get_belief_parameters('f(p2)'))

                    plt.figure(figsize=(15,8))
                    plt.plot(line)
                    plt.xticks(np.arange(len(P1_EVAL_VALUES))[::30], np.round(P1_EVAL_VALUES[::30],1))
                    plt.ylim((0,1.05))
                    plt.ylabel('p')
                    plt.savefig(images_folder + 'sae_p2_{}'.format(counter+1))
                    plt.close()

        message_passing()

        TEST.LOG("EXAMPLE COMPLETE", 1)


