'''
    batch -> insert1 -> module1 -> insert2 -> module2 -> insert3 -> module3 -> insert4 -> module4 -> result
'''

class Batch2Result:
    '''
        batch -> insert1 -> module1 -> insert2 -> module2 -> insert3 -> module3 -> insert4 -> module4 -> result
    '''

    @staticmethod
    def insert1():
        raise NotImplementedError
    
    @staticmethod
    def module1():
        raise NotImplementedError

    @staticmethod
    def insert2():
        raise NotImplementedError

    @staticmethod
    def module2():
        raise NotImplementedError

    @staticmethod    
    def insert3():
        raise NotImplementedError

    @staticmethod
    def module3():
        raise NotImplementedError

    @staticmethod    
    def insert4():
        raise NotImplementedError

    @staticmethod
    def module4():
        raise NotImplementedError