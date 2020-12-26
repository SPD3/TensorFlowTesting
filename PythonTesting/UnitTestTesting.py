import unittest
from Foo import Foo
from UnitTestTesting2 import SimpleTestCaseTwo

class SimpleTestCase(unittest.TestCase):

    def setUp(self):
        """Call before every test case."""
        self.foo = Foo()

    def testGettingOne(self):
        assert self.foo.getTheNumber1() == 1, "getTheNumber1() is not calculating values correctly"
    
    def testGettingTwo(self):
        assert self.foo.getTheNumber2() == 2, "getTheNumber2() is not calculating values correctly"

    def testGettingThree(self):
        assert self.foo.getTheNumber3() == 3, "getTheNumber3() is not calculating values correctly"
    
    def testGettingAString(self):
        assert self.foo.getTheStringHelloWorld() == "Hello World", "Getting hello world is not working"

    def testAssertingTwoObjects(self):
        foo1 = Foo()
        foo2 = Foo()
        foo1.doNothing()
        assert foo1 == foo2, "Uh Oh the foos aren't equal"


if __name__ == "__main__":
    unittest.main()