# 使用type可以直接创建类
Girl = type("Girl",(),{"country":"china","sex":"male"}) # country 属于类的属性


# 还可以创建带有方法的类

## python中方法有普通方法，类方法，静态方法。
def speak(self): #要带有参数self,因为类中方法默认带self参数。
    print("这是给类添加的普通方法")

@classmethod
def c_run(cls):
    print("这是给类添加的类方法")

@staticmethod
def s_eat():
    print("这是给类添加的静态方法")


Boy = type("Boy",(),{"speak":speak,"c_run":c_run,"s_eat":s_eat,"sex":"female"})

# 还可以通过type 定义带 继承的类
class Person(object):
    def __init__(self,name):
        self.name = name
    def p(self):
        print("这是Person的方法")
class Animal(object):
    def run(self):
        print("animal can run ")
#定义一个拥有继承的类，继承的效果和性质和class一样。
Worker = type("Worker",(Person,Animal),{"job":"程序员"})
print(Worker.__class__.__class__) # 查看类的类

# type 的理解
# - 元类就是类的类，python中函数type实际上是一个元类：用于创建元类；
# - type就是Python在背后用来创建所有类的元类;
# - python查看对象所属类型既可以用type函数，也可以用对象自带的__class__属性


# part 2：

# __base__ : 可以查看父类
# __class__ : 查看instance的类型
# object 没有父类，它是所有父类的顶端，但它是由type实例化来的；
# type的父类是 object，但同时：type 是由它自己实例化而来的； type.__base__ == object
#

