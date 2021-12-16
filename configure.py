import sys
import os

#DFTEFE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DFTEFE_PATH = os.environ['DFTEFE_PATH'] 
cmake_dict = {'DFTEFE_PATH': DFTEFE_PATH,
              'DFTEFE_BLAS_LIBRARIES':'',
              'DFTEFE_SCALAPACK_LIBRARIES':'',
              'CMAKE_PREFIX_PATH':'',
              'CMAKE_CXX_FLAGS':'',
              'ENABLE_CUDA':'OFF',
              'CMAKE_CUDA_FLAGS':''}

def sanityCheck(string):
    if '=' not in string:
        raise Exception('''Invalid command line option ''' + string +
                        '''passed''')

def splitString(string):
    if string[:2] != "--":
        raise Exception('''Invalid command line option ''' + string +
                        '''passed. Maybe you forgot to prefix with "--"''')
    pos = string.find('=')
    return string[2:pos], string[pos+1:]

def wrapInDoubleQuotes(string):
    return '"'+ string +'"'


if __name__ == "__main__":
    numArgs = len(sys.argv)
    if numArgs < 2:
        raise Exception('''No options passed to configure. Use python configure.py
                        --help to see the options''')
    
    if sys.argv[1] == '--help' or sys.argv[1] == '-h':
        print('Options')
        print('-------------------------')
        print('--DFTEFE_PATH\t\t\t'+'Path to the parent directory of dft-efe')
        print('\t\t\t\tE.g., --PATH="/home/softwares/dft-efe"\n')
        print('--DFTEFE_BLAS_LIBRARIES\t\t'+'Path to blas libraries')
        print('''\t\t\t\tE.g.,--BLAS_LIBRARIES="-L/path/to/blas/lib/intel64
              \t\t\t\t-Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core
              \t\t\t\t-lgomp -lpthread -lm -ldl"\n''')
        print('--DFTEFE_SCALAPACK_LIBRARIES\t'+'Path to scalapack libraries')
        print('''\t\t\t\tE.g.,--SCALAPACK_LIBRARIES="-L/path/to/scalapack/lib/intel64
              \t\t\t\t-lmkl_scalapack_lp64 -Wl,--no-as-needed -lmkl_intel_lp64
              \t\t\t\t-lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -lgomp
              \t\t\t\t-lpthread -lm -ldl"\n''')
        print('--CMAKE_PREFIX_PATH\t\t'+'Path to the deal.ii installation')
        print('''\t\t\t\tE.g.,--CMAKE_PREFIX_PATH="/path/to/deal.ii"\n''')
        print('--CMAKE_CXX_FLAGS\t\t'+'Flags for C++ compiler')
        print('\t\t\t\tE.g., --CMAKE_CXX_FLAGS="-std=c++11 -march=native"\n')
        print('--ENABLE_CUDA\t\t\t'+'''ON or OFF based on whether to use CUDA/GPU or not''')
        print('--CMAKE_CUDA_FLAGS\t\t'+'Additional flags for CUDA')
        print('\t\t\t\tE.g., --CMAKE_CUDA_FLAGS="-arch=sm_70"\n')
    
    else:
        f = open('CMakeConfigOptions.txt','w')
        for i in range(1,numArgs):
            arg = sys.argv[i]
            sanityCheck(arg)
            key,value = splitString(arg)
            cmake_dict[key] = value

        for key in cmake_dict:
            value = cmake_dict[key]
            if key not in ['ENABLE_CUDA']:
                value = wrapInDoubleQuotes(value)
            
            print(key + '=' +  value, file=f)

        f.close()

