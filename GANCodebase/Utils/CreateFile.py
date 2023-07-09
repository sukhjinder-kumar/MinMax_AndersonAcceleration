import os


def CreateFile(filePath):
    '''
    @Params:
    filePath: string : path to the file
    @Returns:
    filePath is created
    '''
    directory = os.path.dirname(filePath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.isfile(filePath):
        open(filePath, 'w').close()
