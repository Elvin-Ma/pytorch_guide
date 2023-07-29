
from abc import ABCMeta,abstractmethod

class IStream(metaclass=ABCMeta):
    @abstractmethod
    def read(self, maxbytes=-1):
        pass  
    @abstractmethod
    def write(self, data):
        pass
    
class SocketStream(IStream):
    def read(self, maxbytes=-1):
        print("I am in sokerstream:read")
    def write(self, data):
        print("I am in sokerstream:write")
    
if __name__ == "__main__":
    a = SocketStream()
    print(isinstance(a,IStream))
    