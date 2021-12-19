import sys
import os
import textwrap
import traceback

cmake_dict = {'DFTEFE_BLAS_LIBRARIES':['', 'Path to blas libraries',
                                       '''--DFTEFE_BLAS_LIBRARIES=
                                       "-L/path/to/blas/lib/intel64
                                       -Wl,--no-as-needed -lmkl_intel_lp64 
                                       -lmkl_gnu_thread -lmkl_core
                                       -lgomp -lpthread
                                       -lm -ldl"'''],
              'DFTEFE_SCALAPACK_LIBRARIES':['','Path to scalapack libraries',
                                            '''--DFTEFE_SCALAPACK_LIBRARIES=
                                            "-L/path/to/scalapack/lib/intel64
                                            -lmkl_scalapack_lp64 -Wl,--no-as-needed 
                                            -lmkl_intel_lp64 -lmkl_gnu_thread 
                                            -lmkl_core -lmkl_blacs_intelmpi_lp64
                                            -lgomp -lpthread -lm -ldl"'''],
              'DEALII_PATH':['','Path to the deal.ii installation',
                                   '--DEALII_PATH=/path/to/deal.ii/installation'],
              'CMAKE_C_COMPILER':['', 'C compiler to use',
                                  '--CMAKE_C_COMPILER=gcc'],
              'CMAKE_C_FLAGS':['','C compiler flags',
                               '--CMAKE_C_FLAGS="-g -O2"'], 
              'CMAKE_CXX_COMPILER':['', 'C++ compiler to use',
                                  '--CMAKE_CXX_COMPILER=gcc'],
              'CMAKE_CXX_FLAGS':['','C++ compiler flags',
                                 '--CMAKE_CXX_FLAGS="-g -O2"'], 
              'MPI_C_COMPILER':['', 'MPI C compiler to use',
                                  '--MPI_C_COMPILER=mpicc'],
              'MPI_CXX_COMPILER':['', 'MPI C++ compiler to use',
                                  '--MPI_CXX_COMPILER=mpicc++'],
              'ENABLE_CUDA':['OFF','ON or OFF based on whether to use CUDA/GPU or not',
                             '--ENABLE_CUDA=OFF'],
              'CMAKE_CUDA_FLAGS':['','Additional flags for CUDA',
                                  '''--CMAKE_CUDA_FLAGS="-arch=sm_70"''']}

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
            if key not in ['ENABLE_CUDA', 'CMAKE_C_COMPILER',
                           'CMAKE_CXX_COMPILER', 'MPI_C_COMPILER',
                           'MPI_CXX_COMPILER']:
                value = wrapInDoubleQuotes(value)
            
            print(key + '=' +  value, file=f)

        f.close()

