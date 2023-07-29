# coding:utf8

#创建一个新实例时调用__new__，初始化一个实例时用__init__
#在__new__返回一个cls的实例时，后面的__init__才能被调用



class MyClass2():
    ccc = 4


class MyClass():
    abc = 123
    
    def __new__(cls,name):
        print("I am in __new__")        
        obj2 = object.__new__(cls) #要借助父类object的__new__创建对象
        return obj2
    
    def __init__(self,name):
        self.name = name
        print("I am in __init__")
    
#实例化对象
obj = MyClass("monica")
#obj2 = MyClass2("monica")
#print (obj)
#print (obj2.ccc)


