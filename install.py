import CMakeFlagsParser as cmflags
import os
import sys
if __name__ == "__main__":
    numArgs = len(sys.argv)
    if numArgs < 2 or numArgs > 3 :
        raise Exception('''The options to install.py are: 
                        arch [/path/to/build/directory]. arch can be either cpu
                        or gpu. The optional [/path/to/build/directory]
                        specifies where to build the dft-efe executable, by
                        default it creates a build directory inside the
                        dft-efe parent directory''')

    arch = sys.argv[1]
    config_flags=cmflags.getConfig(arch)

    build_dir="./build"
    if numArgs == 3:
        build_dir=sys.argv[2]

    if not os.path.isdir(build_dir): 
        os.mkdir(build_dir)
    
    os.chdir(build_dir)
    print("Building the executable in:", os.getcwd())
    src_dir =""
    if not 'DFTEFE_PATH' in os.environ:
        raise Exception('''DFTEFE_PATH is not set. Please use 'export
            DFTEFE_PATH=/path/to/dft-efe/parent/folder''')
    else:
        src_dir = os.environ['DFTEFE_PATH']

    y = "cmake " + src_dir
    print("CMake options used: ", end="")
    for x in config_flags:
        y = y + " " + x
        print(x, end=" ")
    
    print()
    os.system(y)
    os.system('make -j')



