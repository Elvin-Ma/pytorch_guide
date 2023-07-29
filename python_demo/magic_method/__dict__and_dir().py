# -*- coding: utf-8 -*-

class A(object):
    class_var = 1
    def __init__(self):
        self.name = 'xy'
        self.age = 2

    @property
    def num(self):
        return self.age + 10

    def fun(self):pass
    def static_f():pass
    def class_f(cls):pass

if __name__ == '__main__':
    a = A()
    #print(a.__dict__)
    #print(A.__dict__)
    #print(object.__dict__)
    #dir(A)
    print(dir(a))
