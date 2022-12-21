import os
import cPickle

from message import Message
from factor import Factor


class PGM(object):
    """Object used to represent PGM object"""

    def __init__(self, out_folder):
        # Create output folderpath if it does not exist
        self.out_folder = out_folder
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)

            
    def _get_value(self, key):
        with open(os.path.join(self.out_folder, key + ".node"), 'rb') as file_handler:
            value = cPickle.load(file_handler)

        return value

    
    def _set_value(self, key, value):
        with open(os.path.join(self.out_folder, key + ".node"), 'wb') as file_handler:
            cPickle.dump(value, file_handler)

            
    def add_fnode(self, fnode_name, **fnode_parameters):
        # Create and store prior of fnode
        fnode_prior = Factor(**fnode_parameters)
        fnode_prior.init_parameters()
        self._set_value("prior_" + fnode_name, fnode_prior.parameters)

        # Create and store belief of fnode
        fnode_belief = Factor(**fnode_parameters)
        fnode_belief.init_parameters()
        self._set_value("belief_" + fnode_name, fnode_belief.parameters)

        # Store empty list to be used for connections of fnode
        self._set_value("links_" + fnode_name, [])

        
    def get_prior_parameters(self, fnode_name):
        return self.get_factor_parameters("prior_" + fnode_name)

    
    def get_belief_parameters(self, fnode_name):
        return self.get_factor_parameters("belief_" + fnode_name)

    
    def get_factor_parameters(self, factor_name):
        return self._get_value(factor_name)

    
    def connect_fnodes(self, fnode_name_1, fnode_name_2, domain):
        # Obtain prior parameters of fnodes to be connected
        fnode_parameters_1 = self.get_prior_parameters(fnode_name_1)
        fnode_parameters_2 = self.get_prior_parameters(fnode_name_2)

        # Create message from fnode_1 to fnode_2
        message_1to2_parameters = Message().init_message(
            fnode_parameters_1, fnode_parameters_2,
            domain)
        self._set_value(fnode_name_1 + "->" + fnode_name_2, message_1to2_parameters)

        # Update links for first fnode
        links_1 = self._get_value("links_" + fnode_name_1)
        links_1.append(fnode_name_2)
        self._set_value("links_" + fnode_name_1, links_1)

        # Create message from fnode_2 to fnode_1
        message_2to1_parameters = Message().init_message(
            fnode_parameters_2, fnode_parameters_1,
            domain)
        self._set_value(fnode_name_2 + "->" + fnode_name_1, message_2to1_parameters)

        # Update links for second fnode
        links_2 = self._get_value("links_" + fnode_name_2)
        links_2.append(fnode_name_1)
        self._set_value("links_" + fnode_name_2, links_2)

        
    def get_message_parameters(self, from_fnode_name, to_fnode_name):
        return self._get_value(from_fnode_name + "->" + to_fnode_name)

    
    def update_belief(self, fnode_name):
        # Obtain prior of given fnode
        fnode_belief = Factor(**self._get_value("prior_" + fnode_name))

        # Loop through all connections and update prior through multiplication
        links = self._get_value("links_" + fnode_name)
        for factor in links:
            incoming_message = self.get_message_parameters(factor, fnode_name)
            fnode_belief.multiply_by(Factor(**incoming_message))

        # Store updated belief
        self._set_value("belief_" + fnode_name, fnode_belief.parameters)

        
    def update_message(self, from_fnode_name, to_fnode_name):
        # Obtain beliefs of given fnodes
        from_fnode_parameters = self.get_belief_parameters(from_fnode_name)
        to_fnode_parameters = self.get_belief_parameters(to_fnode_name)

        # Obtain current messages between fnodes
        inc_message_parameters = self.get_message_parameters(
            to_fnode_name, from_fnode_name)
        out_message_parameters = self.get_message_parameters(
            from_fnode_name, to_fnode_name)

        # Estimate new message from between given fnodes
        message_parameters = Message().est_message(
            from_fnode_parameters, to_fnode_parameters,
            inc_message_parameters, out_message_parameters)
        links = self._set_value(from_fnode_name + "->" + to_fnode_name, message_parameters)
