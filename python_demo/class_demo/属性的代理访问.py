# coding:utf8
class A:
    def spam(self, x):
        print("x = ",x)  
    def foo(self):
        print("I am in A:foo")
    
class B:
    """ 使用__getattr__ 的代理，代理方法比较多时候"""
    def __init__(self):
        self._a = A()
    def bar(self):
        pass
    # Expose all of the methods defined on class A
    def __getattr__(self, name):
        """ 这个方法在访问的attribute 不存在的时候被调用
        the __getattr__() method is actually a fallback method
        that only gets called when an attribute is not found"""
        return getattr(self._a, name)   
    
if __name__ == "__main__":
    b = B()
    #b.bar() # Calls B.bar() (exists on B)
    b.spam(1000)
    