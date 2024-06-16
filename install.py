import CMakeFlagsParser as cmflags
import os
import sys
import textwrap
opts_dict={'build_dir' : './build', 'n' : ''}

def getUsageMsg():
    msg = '''The correct usage is python install.py [--n=<numprocs>]\n'''\
          '''[--build_dir=/path/to/build/directory]\n'''\
          '''[--src_dir=/path/to/source/directory]\n'''\
          '''The optional [--build_dir=/path/to/build/directory] specifies '''\
          '''where to build the dft-efe executable.\nBy default it creates '''\
          '''a build directory inside the dft-efe parent directory'''\
          '''The optional [--src_dir=/path/to/source/directory] specifies '''\
          '''where the source files or smore specifically the '''\
          '''CMakeLists.txt exists.\nThe default path is the '''\
          ''' dft-efe parent directory that is set by the DFTEFE_PATH '''\
          '''environment variable.'''
    return msg

def sanityCheck(string):
    if '=' not in string:
        raise Exception('''Invalid command line option ''' + string + '''
                        passed.\nMaybe you forgot "=" to assign the option?\n'''
                        + getUsageMsg())

def splitString(string):
    if string[:2] != "--":
        raise Exception('''Invalid command line option ''' + string +
                        ''' passed.\nMaybe you forgot to prefix with "--"?''')

    pos = string.find('=')
    return string[2:pos], string[pos+1:]

def updateOptsDictFromCommandLine(strings):
    valid_keys = []
    for key in opts_dict:
        valid_keys.append(key)

    for string in strings:
        sanityCheck(string)
        key,value = splitString(string)
        if key not in valid_keys:
            raise Exception('''Invalid options ''' + key + ''' passed. '''\
                            '''Run python install.py --help for info on correct usage''')

        opts_dict[key] = value


if __name__ == "__main__":
    numArgs = len(sys.argv)
    if numArgs > 4:
        raise Exception('''Invalid options passed.\n\n''' +
                        getUsageMsg())

    if numArgs == 3 and (sys.argv[1] == '--help' or sys.argv[1] == '-h'):
        print(getUsageMsg())

    else:
        if not 'DFTEFE_PATH' in os.environ:
            raise Exception('''DFTEFE_PATH is not set. Please use export '''\
                            '''DFTEFE_PATH=/path/to/dft-efe/parent/folder''')
        else:
            opts_dict['src_dir'] = os.environ['DFTEFE_PATH']
        
        updateOptsDictFromCommandLine(sys.argv[1:])
        config_flags = cmflags.getConfig()
        nprocs = opts_dict['n']
        src_dir = opts_dict['src_dir']
        build_dir = opts_dict['build_dir']
        if not os.path.isdir(build_dir):
            os.mkdir(build_dir)

        os.chdir(build_dir)
        print("Building the executable in:", os.getcwd())

        y = "cmake " + src_dir
        print("CMake options used: ", end="")
        for x in config_flags:
            y = y + " " + x
            print(x, end=" ")

        print()
        os.system(y)
        os.system('make -j '+nprocs)
