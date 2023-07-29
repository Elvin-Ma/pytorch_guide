# coding:utf8
# https://www.runoob.com/w3cnote/python-super-detail-intro.html
class A(object):
    def __init__(self):
        self.n=2
    def add(self,m):
        self.n +=m
class B(A):
    def __init__(self):
        self.n=4
    def add(self,m):
        super(B,C).add(self,m)
        self.n +=2 
class C(B):
    def __init__(self):
        self.n=3
    def add(self,m):
        super(C,C).add(self,m) # B 代表什么
        self.n +=3       
        
if __name__ == "__main__":    
    c = C()
    c.add(2)
    print(c.n)
    print(C.__mro__)
