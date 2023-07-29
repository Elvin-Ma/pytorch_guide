# coding:utf8
# -*- coding: UTF-8 -*- 
class A(object):
    m = 10 
    n = 20
    def __init__(self,a,b):
        self.a = a
        self.b = b
        
    def foo(self, x): # 实例方法
        print("executing foo(%s,%s)" % (self, x))
        print('self:', self)
        return self.a
    @classmethod # 类方法
    def class_foo(cls, x):
        print("executing class_foo(%s,%s)" % (cls, x))
        print('cls:', cls)
        return cls.n
    @staticmethod # 静态方法
    def static_foo(x):
        print("executing static_foo(%s)" % x) 
        return    

a = A(3,4) # A 实例化了
b = A(4,5) # B 是例化了
print(a.__dict__)
a.m = 1000 # 新增了一个实例属性，类属性还在

print(a.__dict__)
print(A.__dict__)




