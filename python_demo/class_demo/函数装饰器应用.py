# coding:utf8
import functools

def user_login_data(f):
    @functools.wraps(f)  #可以保持当前装饰器去装饰的函数的 __name__ 的值不变
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper

@user_login_data
def num1():
    print("aaa")



@user_login_data
def num2():
    print("bbbb")

if __name__ == '__main__':
    print(num1.__name__)
    print(num2.__name__)
    
    