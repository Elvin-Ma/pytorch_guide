# coding:utf8
class Base:
    def __init__(self):
        print('Base.__init__')
class A(Base):
    def __init__(self):
        #Base.__init__(self)
        print('A.__init__')
        super().__init__()
       
class B(Base):
    def __init__(self):
        #Base.__init__(self)
        print('B.__init__')
        super().__init__()
        
class C(A,B):
    def __init__(self):
        #A.__init__(self)
        #B.__init__(self)
        print('C.__init__')
        super().__init__()
        
        
if __name__ == "__main__":
    a = C() 
    print(C.__mro__) # 方法解析顺序表
    
