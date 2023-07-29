# coding:utf8

class A(object):
    def __init__(self):
        print("a init")
        self.n=2
    def add(self,m):
        print("A")
        self.n+=m
class B(A):
    def __init__(self):
        print("b init")
        self.n=3
    def add(self,m):
        print("B")
        # D 确定继承顺序
        super(B,D).add(self,m) # mro 表根据D 来确定
        self.n+=3
class C(A):
    def __init__(self):
        print("c init")
        self.n=-1000
    def add(self,m):
        print("C")
        super(C,C).add(self,m)
        self.n+=4
class D(B,C):    
    def __init__(self):
        print("D init")
        self.n= 5
        super(B,self).__init__()
    def add(self,m):
        print("D")
        super(D,D).add(self,m)
        self.n+=5
        
        
if __name__ == "__main__":
    print(D.__mro__)
    print(B.__mro__)
    d=D()
    super(B, D).add(d, 10) # B 找索引的起始位置， D 确定mro 表
    super(B, d).add(10) # d 表示直接传入一个instance，同时确定mro表
    # print(type(D))
    # print(type(d).__mro__)
    # d.add(5)
    #b = B()
    ##d = D()
    ##print(d.n)
    #b.aadd(4)
    ##print(d.n)
    #print(b.n)
    # c = C()
    # print(super(C, c).__init__())
    # print(super(C, c).__init__())




