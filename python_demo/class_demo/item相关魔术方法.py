# coding:utf8
from abc import ABCMeta, abstractmethod

class IStream(metaclass=ABCMeta):
    @abstractmethod
    def read(self, maxbytes=-1):
        print('abs_read')
    @abstractmethod
    def write(self, data):
        print('abs_write')
    
class SocketStream(IStream):
    def read(self, maxbytes=-1):
        print(123)
    #def write(self, data):
        #print(data)

a = SocketStream()
#a.write(10)
#a.read()