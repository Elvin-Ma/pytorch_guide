# coding:utf8


class Foo(object):
    
    def __init__(self,price = 50):
        print('调用__init__方法')
        self.price = price
    
    def __call__(self,n):
        print('调用__call__方法')          
        return self.price * n
    
    def how_much_of_book(self,n):
        print('调用how_much_of_book方法')        
        return self.price * n
    
foo = Foo(40)
print(foo(4))
print(foo.how_much_of_book(4))

