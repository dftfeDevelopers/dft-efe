import sys
import os
import textwrap
import traceback

cmake_dict = {'SLATE_DIR':['', 'Path to SLATE installation',
                                       '--SLATE_DIR=/path/to/SLATE/installation'],
              'BOOST_DIR':['','Path to the boost installation',
                                   '--DEALII_PATH=/path/to/boost/installation'],
              'DEALII_PATH':['','Path to the deal.ii installation',
                                   '--DEALII_PATH=/path/to/deal.ii/installation'],
              'CMAKE_BUILD_TYPE':['', 'Build type Debug/Release',
                                  '--CMAKE_BUILD_TYPE=Debug'],                     
              'CMAKE_C_COMPILER':['', 'C compiler to use',
                                  '--CMAKE_C_COMPILER=gcc'],
              'CMAKE_C_FLAGS':['','C compiler flags',
                               '--CMAKE_C_FLAGS="-g -O2"'], 
              'CMAKE_CXX_COMPILER':['', 'C++ compiler to use',
                                  '--CMAKE_CXX_COMPILER=gcc'],
              'CMAKE_CXX_FLAGS':['','C++ compiler flags',
                                 '--CMAKE_CXX_FLAGS="-g -O2"'], 
              'ENABLE_MPI':['ON','''ON or OFF based on whether to use MPI '''\
                           '''or not. Default=ON''',
                             '--ENABLE_MPI=ON'],
              'ENABLE_MPI_DEVICE_AWARE':['ON','''ON or OFF based on whether '''\
                           '''the MPI library is device or cuda aware. Must have '''\
                           '''--ENABLE_MPI=ON for it to make sense. Default=OFF''',
                             '--ENABLE_MPI_DEVICE_AWARE=OFF'],                             
              'MPI_C_COMPILER':['', '''MPI C compiler to use. Must have
                                --ENABLE_MPI=ON for it to make sense.''',
                                  '--MPI_C_COMPILER=mpicc'],
              'MPI_CXX_COMPILER':['', '''MPI C++ compiler to use. Must have
                                  --ENABLE_MPI=ON for it to make sense.''',
                                  '--MPI_CXX_COMPILER=mpicc++'],
              'ENABLE_CUDA':['OFF','ON or OFF based on whether to use CUDA/GPU or not',
                             '--ENABLE_CUDA=OFF'],
              'CMAKE_CUDA_FLAGS':['','Additional flags for CUDA',
                                  '''--CMAKE_CUDA_FLAGS="-arch=sm_70"'''],
              'LIBXML_LIBRARIES': ['', 'Path to libxml2 libraries',
                                  '''--LIBXML_LIBRARIES=
                                   "-L/path/to/libxml2/libraries -lxml2"
                                  '''],
              'LIBXML_PATH': ['', 'Path to libxml2 include director',
                                  '''--LIBXML_PATH=
                                   "/path/to/libxml2/include"
                                  ''']}

def sanityCheck(string):
    if '=' not in string:
        raise Exception('''Invalid command line option ''' + string + 
                        ''' passed. Maybe you forgot to use "=" to assign'''\
                        '''the option?''')

def splitString(string):
    if string[:2] != "--":
        raise Exception('''Invalid command line option ''' + string + 
                        ''' passed. Maybe you forgot to prefix with "--"?''')
    pos = string.find('=')
    return string[2:pos], string[pos+1:]

def wrapInDoubleQuotes(string):
    return '"'+ string +'"'


if __name__ == "__main__":

    
    numArgs = len(sys.argv)
    if numArgs < 2:
        raise ValueError('''No options passed to configure. Use python ''' \
                             '''configure.py --help to see the options''')

    if sys.argv[1] == '--help' or sys.argv[1] == '-h':
        print('Options')
        print('--------')
        max_key_length = 0
        for key in cmake_dict:
            if(len(key) > max_key_length):
                max_key_length = len(key)

        for key in cmake_dict:
            value = cmake_dict[key]
            print("--{0:<{1}}\t{2}".format(key,max_key_length,value[1]))
            print("\t"+' '*(max_key_length-2)+'e.g.,'+value[2]+"\n")
    
    else:
        if not 'DFTEFE_PATH' in os.environ:
            raise KeyError('''DFTEFE_PATH is not set. Please use export'''\
                 '''DFTEFE_PATH=/path/to/dft-efe/parent/folder''')
        else:
            DFTEFE_PATH = os.environ['DFTEFE_PATH']

        f = open('CMakeConfigOptions.txt','w')
        print('DFTEFE_PATH='+ '"' + DFTEFE_PATH + '"', file=f)
        for i in range(1,numArgs):
            arg = sys.argv[i]
            sanityCheck(arg)
            key,value = splitString(arg)
            cmake_dict[key][0] = value


        for key in cmake_dict:
            value = cmake_dict[key][0]
            if key not in ['ENABLE_MPI', 'ENABLE_MPI_DEVICE_AWARE', 'ENABLE_CUDA', 'CMAKE_BUILD_TYPE', 'CMAKE_C_COMPILER',
                           'CMAKE_CXX_COMPILER', 'MPI_C_COMPILER',
                           'MPI_CXX_COMPILER']:
                value = wrapInDoubleQuotes(value)
            
            print(key + '=' +  value, file=f)

        f.close()

