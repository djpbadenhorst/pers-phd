# Class to be used to store and react on unit test variables
class TestVariables(object):
    def __init__(self):
        """Initialize TestVariables object"""
        
        self.log_level = 3

    def LOG(self, log_message, log_level):
        """Log given message
        Args :
            log_message (str) :
                Message to be logged
            log_level (int) :
                Level at which logging should happen
        """

        if log_level <= self.log_level:
            print(log_message)

    def EQUALS(self, test_case, object_1, object_2, error_message, verbose=False):
        """
        Args :
            object_1 (NA):
                First object in comparison
            object_2 (NA):
                Second object in comparison
            error_message (str) :
                Error message when objects are not equal
            verbose (bool) :
                Allow additional outputs for debugging
        """

        from numpy import ndarray

        if verbose:
            print 'Test equality'
            print str(object_1)
            print str(object_2)

        try:
            if isinstance(object_1, (int, float, long, complex)):
                test_case.assertAlmostEqual(object_1, object_2, 5, error_message)
                    
            elif isinstance(object_1, (list, tuple, ndarray)):
                test_case.assertEqual(len(object_1), len(object_2))
                for index in xrange(len(object_1)):
                    var1, var2 = object_1[index], object_2[index]
                    self.EQUALS(test_case, var1, var2, error_message, verbose)
                    
            elif isinstance(object_1, dict):
                test_case.assertEqual(set(object_1), set(object_2))
                for key in object_1:
                    self.EQUALS(test_case, object_1[key], object_2[key], error_message, verbose)
                    
            else:
                test_case.assertEqual(object_1, object_2)
                
        except Exception as tmp_error:
            raise Exception(error_message)

        
# Class to be used to store and react on common variables
class CommonVariables(object):
    def __init__(self):
        """Initialize CommonVariables object"""
        
        self.error_check = False
        
        
# Create global variable containing unit test variables
global TEST

# Create global TestVariables object
TEST = TestVariables()

# Create global variable containing common variables
global COMMON

# Create global CommonVariables object
COMMON = CommonVariables()
