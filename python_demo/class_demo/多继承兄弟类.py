# coding:utf8
#coding:utf-8
#单继承
class A(object):
    def __init__(self):
        self.n=2
    def add(self,m):
        self.n +=m
class B(A):
    def __init__(self):
        self.n=3
    def add(self,m):
        #super(B,self).add(m)
        self.n +=3
b=B()
b.add(3)
print(b.n)
