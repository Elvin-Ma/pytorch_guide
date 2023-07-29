# coding:utf8

class nnn:
    
    def __init__(self,name):
        self.name = name

    def __enter__(self):
        print('出现with语句,对象的__enter__被触发,有返回值则赋值给as声明的变量')
        return self

    def __exit__(self, exc_type=0, exc_val=1, exc_tb=2):
        print('with中代码块执行完毕时执行我啊')

        #return True

if __name__ == "__main__":
    
    with nnn('Monica') as f:
        
        print(f)
        print(f.name)
