"""
子程序：子程序，或者称为函数，在所有语言中都是层级调用，
比如A调用B，B在执行过程中又调用了C，
C执行完毕返回，B执行完毕返回，最后是A执行完毕。
子程序调用总是一个入口，一次返回，调用顺序是明确的。而协程的调用和子程序不同。
协程：协程看上去也是子程序，但执行过程中，在子程序内部可中断，
然后转而执行别的子程序，在适当的时候再返回来接着执行。
"""
def consumer():
    r = 100
    while True:
        n = yield r
        if not n:
            return
        print('[CONSUMER] Consuming %s...' % n)
        r = '200 OK'

def produce(c):
    d = c.send(None)
    n = 0
    while n < 5:
        n = n + 1
        print('[PRODUCER] Producing %s...' % n)
        r = c.send(n)
        print('[PRODUCER] Consumer return: %s' % r)
    c.close()

if __name__ == "__main__":
  c = consumer()
  produce(c)
  print("run corutine.py successfully !!!")
