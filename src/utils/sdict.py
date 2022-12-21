class SDict(object):
    """Object used as wrapper on standard dict object"""
    
    def __init__(self, **attributes):
        """Initialization of SDict object

        Arguments : 
            attributes (dict) :
                Set of key value pairs to be used in SDict object
        """

        self.attributes = attributes

    def __str__(self):
        """Convert SDict object to string"""

        return self.__indent_str__(1)
    

    def __indent_str__(self, indent_level):
        """Create indentend string representation of SDict object
        Arguments : 
            indent_level (int) :
                Indicates the indentation level to be used
        """

        # Create string to be used for indentation
        indent = ("  ")*indent_level

        # Extract keys and sort them alphabetically
        keys = self.attributes.keys()
        keys.sort()

        # Create return string and add components individually
        ret_var = ""
        for key in keys:
            ret_var += indent + "key : " + key + " => \n"
            for line in self.attributes[key].__str__().splitlines():
                ret_var += indent + "  " + line + "\n"

        return ret_var
