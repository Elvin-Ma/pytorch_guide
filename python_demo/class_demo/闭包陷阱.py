# coding:utf8
#希望一次返回3个函数，分别计算1x1,2x2,3x3:  
def count():  
    fs = []  
    for i in range(1, 4):  
        def f(j):  
            def g():  
                return j*j  
            return g  
        fs.append(f(i))  
    return fs  

f1, f2, f3 = count()  

print(f1(),f2(),f3())

