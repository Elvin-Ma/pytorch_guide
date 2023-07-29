# coding:utf8
def outer(some_func):
    def inner():
        print ("before some_func")
        ret = some_func() # 1
        return ret + 1
    return inner
def foo():
    return 1

decorated = outer(foo) # 2
print(decorated())



