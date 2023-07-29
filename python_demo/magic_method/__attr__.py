#coding:utf8

class A(object):
    # test the __attr__ magic method
    def __init__(self,age):
        self.name = "Bob"
        self.age = age
        self.gender  = "male"
    
    def __getattr__(self,name):
        print("I am in __getattr__ !")

    def __getattribute__(self, attr):
        # 拦截age属性
        if attr == "age":
            print("问年龄是不礼貌的行为")            
        # 非age属性执行默认操作
        else:
            return super().__getattribute__(attr) 
            #return object.__getattribute__(self, attr)
    def __setattr__(self, name, value):
        print("I am in __setattr__")
        return object.__setattr__(self, name, value)
    def __delattr__(self,name):
        print("I am in __delattr__")


if __name__ == "__main__":
    a = A(67)
    print("age = ",a.name)
    getattr(a,"age")
    print(a.y)
    print(a.__dict__)
    print(A.__dict__)
    delattr(a,"age")
    print(a.age)
    #print(age)
    #print(a.name)
    #print(a.gender)
