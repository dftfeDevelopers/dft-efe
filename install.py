import CMakeFlagsParser as cmflags
import os
import sys
import textwrap
opts_dict={'build_dir' : './build', 'n' : '', 'generator' : 'make'}

# The build systems install.py knows how to drive:
#   key -> (CMake generator name, build command)
generators_dict={'make'  : ('Unix Makefiles', 'make'),
                 'ninja' : ('Ninja', 'ninja')}

def getUsageMsg():
    msg = '''The correct usage is python install.py [--n=<numprocs>]\n'''\
          '''[--build_dir=/path/to/build/directory]\n'''\
          '''[--src_dir=/path/to/source/directory]\n'''\
          '''[--generator=make/ninja]\n'''\
          '''The optional [--build_dir=/path/to/build/directory] specifies '''\
          '''where to build the dft-efe executable.\nBy default it creates '''\
          '''a build directory inside the dft-efe parent directory'''\
          '''The optional [--src_dir=/path/to/source/directory] specifies '''\
          '''where the source files or smore specifically the '''\
          '''CMakeLists.txt exists.\nThe default path is the '''\
          ''' dft-efe parent directory that is set by the DFTEFE_PATH '''\
          '''environment variable.\n'''\
          '''The optional [--generator=make/ninja] selects the build system '''\
          '''to generate for and build with.\nThe default is make (i.e., '''\
          '''the "Unix Makefiles" CMake generator). Use --generator=ninja '''\
          '''to build with ninja instead.\nNote that a build directory that '''\
          '''was already configured for one generator cannot be reused for '''\
          '''the other - delete it first.'''
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

def getCachedGenerator(build_dir):
    '''Return the generator a build directory was previously configured with,
       or None if it has not been configured yet.'''
    cache_file = os.path.join(build_dir, 'CMakeCache.txt')
    if not os.path.isfile(cache_file):
        return None

    with open(cache_file, 'r') as f:
        for line in f:
            if line.startswith('CMAKE_GENERATOR:INTERNAL='):
                return line.split('=', 1)[1].strip()

    return None

def run(command):
    print(command)
    status = os.system(command)
    if status != 0:
        sys.exit('''Command failed: ''' + command)


if __name__ == "__main__":
    numArgs = len(sys.argv)
    if numArgs > 5:
        raise Exception('''Invalid options passed.\n\n''' +
                        getUsageMsg())

    if numArgs >= 2 and (sys.argv[1] == '--help' or sys.argv[1] == '-h'):
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
        generator = opts_dict['generator'].lower()
        if generator not in generators_dict:
            raise Exception('''Invalid generator ''' + opts_dict['generator'] +
                            ''' passed. Valid options are: ''' +
                            '/'.join(sorted(generators_dict)))

        cmake_generator, build_command = generators_dict[generator]

        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)

        # CMake cannot switch the generator of an already configured build
        # directory, so fail early with a message that says what to do.
        cached_generator = getCachedGenerator(build_dir)
        if cached_generator is not None and cached_generator != cmake_generator:
            raise Exception(build_dir + ''' was already configured with the "'''
                            + cached_generator + '''" generator, which cannot '''
                            '''be switched to "''' + cmake_generator +
                            '''".\nDelete the build directory (rm -rf ''' +
                            build_dir + ''') and rerun install.py, or pass a '''
                            '''different --build_dir.''')

        os.chdir(build_dir)
        print("Building the executable in:", os.getcwd())

        y = "cmake -G \"" + cmake_generator + "\" " + src_dir
        print("CMake options used: ", end="")
        for x in config_flags:
            y = y + " " + x
            print(x, end=" ")

        print()
        run(y)

        # make with an empty -j means "unlimited jobs" (the historical
        # behaviour of this script when --n is not given), whereas ninja
        # requires an argument after -j, so leave it off and let ninja pick
        # its own default.
        if nprocs:
            build_command = build_command + ' -j ' + nprocs
        elif generator == 'make':
            build_command = build_command + ' -j'

        run(build_command)
