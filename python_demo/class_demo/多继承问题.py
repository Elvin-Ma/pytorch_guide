# coding:utf8
class Base:
    def __init__(self):
        print('Base.__init__')
class A(Base):
    def __init__(self):
        #super().__init__()
        print('A.__init__')
class B(Base):
    def __init__(self):
        super().__init__()
        print('B.__init__')

class C(A,B):
    def __init__(self):
        super().__init__() #一句话即可
        print('C.__init__')
a = C()
print(C.__mro__)