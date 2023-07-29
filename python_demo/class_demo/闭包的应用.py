# coding:utf8

def outer():
    x = 1
    def inner():
        print(x) # 1
    inner() # 2

def outer2():
    x = 1
    y = 0
    def inner():
        print(x) # 1
        print(y)
    return inner

class Averager():

    def __init__(self):
        self.series=[]
    def __call__(self,new_value):
        self.series.append(new_value)
        total = sum(self.series)
        return total/len(self.series)    　

if __name__ == '__main__':
    
    outer()
    foo = outer2()
    foo() # 能打印出1吗？
    print(foo.__closure__)   
    

    avg = Averager()
    print(avg(10))
    print(avg(11))

