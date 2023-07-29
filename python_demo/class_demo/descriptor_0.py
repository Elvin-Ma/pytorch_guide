# coding:utf8
class Integer:
    def __init__(self, name):
        self.name = name
    def __get__(self, instance, cls):
        if instance is None:
            print("I am Integer.")
            return self
        else:
            return instance.__dict__[self.name]
    def __set__(self, instance, value):
        if instance is None:
            print(100000)
            return
        if not isinstance(value, int):
            raise TypeError('Expected an int')
        # 关键一步，将value 加入到 实例的__dict__中
        # operate the instance's dic
        instance.__dict__[self.name] = value
        
    def __delete__(self, instance):
        del instance.__dict__[self.name]

class Point:
    a = 100
    x = Integer('x')
    y = Integer('y')
    def __init__(self, x, y):
        self.x = x  # not simple assign value
        self.y = y        

if __name__ == "__main__":
    test = Integer("Micheal")    
    a = Point(2,3)    
    print(a.x)
    print(Point.x)
    