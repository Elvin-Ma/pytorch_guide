"""
每启动一个进程，都要独立分配资源和拷贝访问的数据，
所以进程的启动和销毁的代价是比较大了，
所以在实际中使用多进程，要根据服务器的配置来设定。
"""
def multi_process_demo1():
  from multiprocessing import  Process
  from time import sleep
  import time

  def fun1(name, index):
      sleep(2)
      print('测试%s多进程: %d' %(name, index))

  process_list = []
  a = time.time()
  for i in range(5):  #开启5个子进程执行fun1函数
      p = Process(target=fun1,args=('Python====', i)) #实例化进程对象
      p.start()
      process_list.append(p)

  for i in process_list:
      p.join()

  print(f'结束测试, 使用时间： {time.time() - a}.')


def multi_process_demo2():
  from multiprocessing import  Process

  class MyProcess(Process): #继承Process类
      def __init__(self,name):
          super(MyProcess,self).__init__()
          self.name = name

      def run(self):
          print('测试%s多进程' % self.name)

  process_list = []
  for i in range(5):  #开启5个子进程执行fun1函数
      p = MyProcess('Python') #实例化进程对象
      p.start()
      process_list.append(p)

  for i in process_list:
      p.join()

  print('结束测试')

def multi_process_queue():
  from multiprocessing import Process,Queue
  def fun1(q,i):
    print('子进程%s 开始put数据' %i)
    q.put('我是%s 通过Queue通信' %i)

  q = Queue()

  process_list = []
  for i in range(3):
      #注意args里面要把q对象传给我们要执行的方法，这样子进程才能和主进程用Queue来通信
      p = Process(target=fun1,args=(q,i,))
      p.start()
      process_list.append(p)

  for i in process_list:
      p.join()

  print('主进程获取Queue数据')
  print("======", q.get())
  print("======", q.get())
  print("======", q.get())
  print('结束测试')

if __name__ == '__main__':
  # multi_process_demo1()
  multi_process_queue()
  print("run multi_process successfully !!!")

