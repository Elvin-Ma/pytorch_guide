
from  multiprocessing import  Process

def fun1(name):
    print('测试%s多进程' %name)

if __name__ == '__main__':
     process_list = []
     for i in range(5):  #开启5个子进程执行fun1函数
          p = Process(target=fun1,args=('Python',)) #实例化进程对象
          p.start()
          process_list.append(p)

     for i in process_list:
          p.join()
          print('结束测试')


