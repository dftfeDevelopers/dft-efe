import os
def sanityCheck(string):
    if '=' not in string:
        raise Exception('''Invalid command line option''' + string +
                        '''passed''')

def splitString(string):
    pos = string.find('=')
    return string[:pos], string[pos+1:]


def getConfig():
    DFTEFE_PATH = os.environ['DFTEFE_PATH']
    f = open(DFTEFE_PATH+'/CMakeConfigOptions.txt', 'r')
    lines = f.readlines()
    if len(lines) == 0:
        raise Exception('''Empty CMakeConfigOptions.txt found. Please run
                        ./configure.py with the correct options''' ) 
    
    cmake_config_opts = []
    cmake_dict = {}
    for line in lines:
        sanityCheck(line)
        key,value = splitString(line)
        cmake_dict[key]=value

    for key in cmake_dict:
        value = cmake_dict[key]
        opts='-D' + key + '=' + value.strip() 
        cmake_config_opts.append(opts)
  
    return cmake_config_opts
