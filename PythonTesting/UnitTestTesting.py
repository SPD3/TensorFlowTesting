import unittest
from Foo import Foo

class SimpleTestCase(unittest.TestCase):

    def setUp(self):
        """Call before every test case."""
        self.foo = Foo()

    def testA(self):
        assert self.foo.getTheNumber1() == 1, "getTheNumber1() is not calculating values correctly"
    
    def testB(self):
        assert self.foo.getTheNumber2() == 2, "getTheNumber2() is not calculating values correctly"

    def testC(self):
        assert self.foo.getTheNumber3() == 3, "getTheNumber3() is not calculating values correctly"
        

if __name__ == "__main__":
    unittest.main()