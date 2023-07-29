
# coding:utf8
import time
class Date:
    """ 方法一：使用类方法"""
    # Primary constructor
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
        
        
    @classmethod
    def today(cls):
        t = time.localtime()
        return cls(t.tm_year, t.tm_mon, t.tm_mday,t.tm_hour) 
    
if __name__ == "__main__":
    a = Date(2012, 12, 21) # Primary
    b = Date.today() # Alternate
    print("today is {}-{}-{}".format(b.year, b.month, b.day))
    
    print(time.time())
    print(time.localtime())
    print(time.strftime('%Y-%m-%d %H:%M:%S'))    