class MyProperty(object):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        print("====>in __get__")
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError
        return self.fget(obj)

    def __set__(self, obj, value):
        print("====>in __set__")
        if self.fset is None:
            raise AttributeError
        self.fset(obj, value)

    def __delete__(self, obj):
        print("====>in __delete__")
        if self.fdel is None:
            raise AttributeError
        self.fdel(obj)

    def getter(self, fget):
        pass

    def setter(self, fset):
        print("====>in setter")
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

    def deleter(self, fdel):
        print("====>in deleter")
        return type(self)(self.fget, self.fset, fdel, self.__doc__)

class Item:
    @MyProperty
    def price(self):
        return self._price

    @price.setter
    def price(self, value):
        if value > 0:
            self._price = value
        else:
            raise ValueError("Valid value..")

def func1():
  item = Item()
  item.price = 100
  print(item.price)

if __name__ == "__main__":
  func1()
  print("run property_demo.py successfully !!!")