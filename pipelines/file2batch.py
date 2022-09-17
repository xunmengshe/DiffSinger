'''
    file -> temporary_dict -> processed_input -> batch
'''

class File2Batch:
    '''
        file -> temporary_dict -> processed_input -> batch
    '''

    @staticmethod
    def file2temporary_dict():
        raise NotImplemented

    @staticmethod
    def temporary_dict2processed_input():
        raise NotImplementedError
    
    @staticmethod
    def processed_input2batch():
        raise NotImplementedError
