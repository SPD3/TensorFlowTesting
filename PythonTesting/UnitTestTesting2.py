import unittest
from Foo import Foo

class SimpleTestCaseTwo(unittest.TestCase):

    def setUp(self):
        """Call before every test case."""
        self.foo = Foo()

    def testGettingFour(self):
        assert self.foo.getTheNumber4() == 4, "getTheNumber4() is not calculating values correctly"
    
    def testGettingFive(self):
        assert self.foo.getTheNumber5() == 5, "getTheNumber2() is not calculating values correctly"
