class Foo:
    def __init__(self):
        self.num = 1

    def getTheNumber1(self):
        return 1

    def getTheNumber2(self):
        return 2

    def getTheNumber3(self):
        return 3

    def getTheNumber4(self):
        return 4

    def getTheNumber5(self):
        return 5

    def getTheStringHelloWorld(self):
        return "Hello World"

    def doNothing(self):
        return

    def __eq__(self, value):
        return self.num == value.num

    