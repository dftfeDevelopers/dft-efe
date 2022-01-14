import reframe as rfm
import reframe.utility.sanity as sn
from reframe.utility.sanity import evaluate
from reframe.core.backends import getlauncher
import os
DFTEFE_PATH=''
if not 'DFTEFE_PATH' in os.environ:
    raise Exception('''DFTEFE_PATH is not set. Please use export'''\
                    '''DFTEFE_PATH=/path/to/dft-efe/parent/folder''')
else:
    DFTEFE_PATH = os.environ['DFTEFE_PATH']

parser = rfm.utility.import_module_from_file(DFTEFE_PATH+"/test/Parser.py")
cu = rfm.utility.import_module_from_file(DFTEFE_PATH+"/test/CompareUtil.py")
ss = rfm.utility.import_module_from_file(DFTEFE_PATH+"/test/SetupSystems.py")
bincpy = rfm.utility.import_module_from_file(DFTEFE_PATH+"/test/BinaryCopier.py")
cmflags = rfm.utility.import_module_from_file(DFTEFE_PATH+"/CMakeFlagsParser.py")

"""
Types of tags
--------------
compile: Make only tests
run: Make and run tests
unit: Tests that related to only a single function
aggregate: Tests for a complex function
slow: Tests requiring more than 30s of wall time
fast: Tests requiring less than 30s of wall time
cpu: Tests to be run on cpu only
gpu: Tests to be run on gpu only
both: Tests to be run on  both cpu and gpu
serial: Serial tests (ideally requires no mpi or openmp)
parallel: Parallel tests that requires mpi or openmp
"""
##########################################################
############## Example of a Build Only Test ###############
##########################################################
@rfm.simple_test
class BuildOnly(rfm.CompileOnlyRegressionTest):
    descr = 'A build only test using CMake'
    build_system = 'CMake'
    # Provide the target that CMake needs to make. By default it builds
    # all the targets. In practice, a single CMakeLists might contain targets
    # for multiple different tests, whereas the test at hand might require building
    # only one of the targets. An empty list makes all the targets
    make_opts = []
    builddir ='./build'
    
    # Path to the source directory containing the CMakeLists.txt
    # The relative path is from the directory of this file
    sourcesdir = './src'
    
    # As a standard convention, we use 4 categories of tags that can help 
    # us run only tests matching certain tag(s). See the top of this file 
    # for the description on types. A user should populate the tagsDict with 
    # the appropriate values for each of the four keys: 'compileOrRun',
    # 'unitOrAggregate', 'arch', 'serialOrParallel'
    tagsDict = {'compileOrRun': 'compile', 'unitOrAggregate':
                'unit', 'slowOrFast': 'fast', 'arch': 'cpu',
                'serialOrParallel': 'serial'}
    tags = {x.lower() for x in tagsDict.values()}

    # Define valid_systems as a list of strings of the format
    # 'system:partiion'. Over here, based on the 'arch' 
    # (i.e,,'cpu', 'gpu', or 'both'), we filter out
    # the system:partitions from a list named system_partition_list
    # in SetupSystems.py. The convention is for 'arch'='cpu', we
    # select only those system:partitions that do not contain the string 'gpu' 
    # in it. Conversely, for 'arch'='gpu', only those system:partitions are
    # included which contains the string 'gpu'. For 'arch'='both', all the
    # system:partiions are selected.
    # NOTE: For any new systems:partition added to the config file, 
    # they must also be added to the system_partition in setupSystems.py

    # valid_systems = ss.getValidSystems(tagsDict['arch'])
    # valid_prog_environs = ['*']
    valid_systems = ['greatlakes:login']
    valid_prog_environs = ['builtin']
    
    # List of files to save into the test's output folder. 
    # By default, ReFrame deletes all the files stage directory of test,
    # if the test is successful
    keep_files = []
    
    # List of options to pass to CMake. This is where we can
    # pass the various dependent library paths, compilers, etc.
    # e.g., for setting a path to a dependent library (say mkl) 
    # in the variable MKL_DIR that is needed by CMake, you can set
    # config_opts = ['-DMKL_DIR=/path/to/mkl/']
    # In our case, we have written a python script CMakeFlagsParser.py
    # in the main directory of dft-efe that directly reads the config
    # options used to compile the main into passes them to 
    # ReFrame's config_opts. The usage is
    # CMakeFlagsParser.getconfig(archtype), where archtype is a string
    # that defines the archicture type as 'cpu', 'cpu', or 'both'
    # In the following line, CMakeFlagsParser has been aliased into cmflags 
    # (see top of this file)
    config_opts = cmflags.getConfig(tagsDict['arch'])

    @run_before('compile')
    def set_compiler_flags(self):
        self.build_system.make_opts = self.make_opts
        self.build_system.config_opts = self.config_opts
        ''' Set any compilation flag, configuration, environment variables, etc'''
        # NOTE: In most cases, setting the following attribues within a test should be avoided, 
        # it should be set within the config file. 
        # + `self.build_system.cc` string to specify the C compiler
        # + `self.build_system.cflags` list of string to specify the C compiler flags
        # + `self.build_system.cxx` string to specify the C++ compiler
        # + `self.build_system.cxxflags` list of string to specify the C++ compiler flags
        # + `self.build_system.cppflags` list of string to specify the preprocessor flags
        # + `self.build_system.ldflags` list of string to specify the linker flags
        # + `self.build_system.nvcc` string to specify the CUDA compiler
        # + `self.variables` a dictionary to set environment variables. e.g, 
        #  self.variables = {'DFT_EFE_LINKER': '"-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"'}
        # Any special configuration that needs to be done externally can be done
        # using the prebuild_cmds attribute 
        # e.g., self.prebuild_cmds = ['./custom_configure -with-mylib']
        # Avoid using it as far as possible, in most cases one can invoke the
        # same configuration through CMake


    @sanity_function
    def validate_test(self):
        hasWarning = True
        hasError = True
        msgWarning = "Found warning(s) while compiling."
        msgError = "Found error(s) while compiling."
        matches = evaluate(sn.findall(r'warning/i', evaluate(self.stdout)))
        if len(matches) == 0:
            hasWarning = False

        matchesOut = evaluate(sn.findall(r'error/i', evaluate(self.stdout)))
        matchesErr = evaluate(sn.findall(r'error/i', evaluate(self.stderr)))
        if len(matchesOut) == 0 and len(matchesErr) == 0:
            hasError = False
        
        hasTestPassed = not hasWarning and not hasError
        hasTestPassed = False
        msg = ""
        bincpy.BinCpy(os.path.dirname(os.path.abspath(__file__)))
        if hasError:
            msg = msgError

        elif hasWarning:
            msg = msgWarning

        else:
            msg = ""
            return sn.assert_true(hasTestPassed, msg=msg)