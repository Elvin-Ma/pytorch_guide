# coding:utf8
class MyDict(object):

    def __init__(self):
        print('call fun __init__')
        self.item = {}

    def __getitem__(self,key):
        print('call fun __getItem__')
        return self.item.get(key)

    def __setitem__(self,key,value):
        print('call fun __setItem__')
        self.item[key] =value

    def __delitem__(self,key):
        print('cal fun __delitem__')
        del self.item[key]

    def __len__(self):
        return len(self.item)
    
myDict = MyDict()
print (myDict.item)
myDict[2] = 'ch'
myDict['hobb'] = 'sing'
print(myDict.item)

