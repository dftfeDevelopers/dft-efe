import reframe as rfm
import reframe.utility.sanity as sn
from reframe.utility.sanity import evaluate
#import CompareUtil as cu
parser = rfm.utility.import_module_from_file("Parser.py")
cu = rfm.utility.import_module_from_file("CompareUtil.py")
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
serial: Serial tests (ideally requires no mpi or openmp)
parallel: Parallel tests that requires mpi or openmp
"""

@rfm.simple_test
class MakeOnlyTest(rfm.CompileOnlyRegressionTest):
    descr = 'Compile one test using CMake'
    valid_systems = ['*']
    valid_prog_environs = ['*']
    build_system = 'CMake'
    make_opts = ['all']
    builddir ='./build'
    sourcesdir = './src'
    # As a standard convention, we use 4 categories of tags that can help 
    # us run only tests matching certain tag(s). See the top of this file 
    # for the description on types. A user should populate the tagsDict with 
    # the appropriate values for each of the four keys: 'compileOrRun',
    # 'unitOrAggregate', 'cpuOrgpu', 'serialOrParallel'
    tagsDict = {'compileOrRun': 'compile', 'unitOrAggregate':
                'unit', 'slowOrFast': 'fast', 'cpuOrgpu': 'cpu',
                'serialOrParallel': 'serial'}
    tags = {x for x in tagsDict.values()}


    @run_before('compile')
    def set_compiler_flags(self):
        self.build_system.make_opts = self.make_opts
        ''' Set any compilation flag, configuration, environment variables, etc'''
        #NOTE: In most cases, setting the following attribues within a test should be avoided, 
        #it should be set within the config file. 
        #+ `build_system.cc` string to specify the C compiler
	    #+ `build_system.cflags` list of string to specify the C compiler flags
	    #+ `build_system.cxx` string to specify the C++ compiler
	    #+ `build_system.cxxflags` list of string to specify the C++ compiler flags
	    #+ `build_system.cppflags` list of string to specify the preprocessor flags
	    #+ `build_system.ldflags` list of string to specify the linker flags
	    #+ `build_system.nvcc` string to specify the CUDA compiler
        #+ `variables` a dictionary to set environment variables. e.g, 
        #  self.variables = {'DFT_EFE_LINKER': '"-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"'}
        # Any special configuration that needs to be done externally can be done
        # using the prebuild_cmds attribute 
        # e.g., self.prebuild_cmds = ['./custom_configure -with-mylib']
        # Avoid using it as far as possible, in most cases one can invoke the
        # same configuration through CMake


    @sanity_function
    def set_sanity_patterns(self):
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
        msg = msgWarning + msgError
        return sn.assert_true(hasTestPassed, msg=msg)




@rfm.simple_test
class FileOutTest(rfm.RegressionTest):
    descr = 'Regression Test using CMake and output to a file'
    valid_systems = ['*']
    valid_prog_environs = ['*']
    build_system = 'CMake'
    # Provide the target that CMake needs to make. By default it builds
    # all the targets. In practice, a single CMakeLists might contain targets
    # multiple different tests, whereas the test at hand might require building
    # only one of the targets
    make_opts = ['test1']
    #NOTE: Need to specify the name of the executable, as
    # ReFrame has no way of knowing that while building from CMake
    executable = './test1'
    builddir ='./build'
    sourcesdir = './src'
    # As a standard convention, we use 4 categories of tags that can help 
    # us run only tests matching certain tag(s). See the top of this file 
    # for the description on types. A user should populate the tagsDict with 
    # the appropriate values for each of the four keys: 'compileOrRun',
    # 'unitOrAggregate', 'cpuOrgpu', 'serialOrParallel'
    tagsDict = {'compileOrRun': 'compile', 'unitOrAggregate':
                'unit','slowOrFast': 'fast', 'cpuOrgpu': 'cpu',
                'serialOrParallel': 'serial'}
    tags = {x for x in tagsDict.values()}
    # By default ReFrame deletes all the output files generated by the test
    # In case you want to retain any of the files in the output folder of the
    # test, list them in keep_files attribute
    keep_files = ['out_test1', 'out_test2', 'out_test3']

    @run_before('compile')
    def set_compiler_flags(self):
        ''' Set any compilation flag, configuration, environment variables, etc'''
        # set the make_opts as defined in the Constructor 
        self.build_system.make_opts = self.make_opts
        #NOTE: In most cases, setting the following attribues within a test should be avoided, 
        #it should be set within the config file. 
        #+ `build_system.cc` string to specify the C compiler
	    #+ `build_system.cflags` list of string to specify the C compiler flags
	    #+ `build_system.cxx` string to specify the C++ compiler
	    #+ `build_system.cxxflags` list of string to specify the C++ compiler flags
	    #+ `build_system.cppflags` list of string to specify the preprocessor flags
	    #+ `build_system.ldflags` list of string to specify the linker flags
	    #+ `build_system.nvcc` string to specify the CUDA compiler
        #+ `variables` a dictionary to set environment variables. e.g, 
        #  self.variables = {'DFT_EFE_LINKER': '"-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"'}
        # Any special configuration that needs to be done externally can be done
        # using the prebuild_cmds attribute 
        # e.g., self.prebuild_cmds = ['./custom_configure -with-mylib']
        # Avoid using it as far as possible, in most cases one can invoke the
        # same configuration through CMake

    @sanity_function
    def validate_test(self):
        '''
        A user-defined function which decides whether the test passed or not as
        well as define an error message to display if the test fails.
        '''
        #NOTE: To make the tests portable, we do not hardcode any values or
        #error message in this file. Everything must be read from the output of
        #the test (either by catching the stdout or by reading from an output
        #file). The error message to display must be specified in this function
        #(or by invoking some other function from here). To integrate with
        #ReFrame's error handling and logging, we pass the boolean storing 
        #whether the test passed or not and the error message to ReFrame's
        # reframe.utility.sanity.assert_true(hasTestPassed, msg) function
        hasTestPassed = True
        msg = 'Failed for some reason'
        bmfilename = "benchmark_test1"
        outfilename = "out_test1"
        bmParser = parser.Parser.fromFilename(bmfilename)
        outParser = parser.Parser.fromFilename(outfilename)
        bmVal = bmParser.extractKeyValues("Values")
        outVal = outParser.extractKeyValues("Values")
        hasTestPassed, msg = cu.Compare().cmp(bmVal[0], outVal[0])
        return sn.assert_true(hasTestPassed, msg=msg)

@rfm.simple_test
class StdOutTest(rfm.RegressionTest):
    descr = 'Regression Test using CMake and stdout'
    valid_systems = ['*']
    valid_prog_environs = ['*']
    build_system = 'CMake'
    # Provide the target that CMake needs to make. By default it builds
    # all the targets. In practice, a single CMakeLists might contain targets
    # multiple different tests, whereas the test at hand might require building
    # only one of the targets
    make_opts = ['test2']
    #NOTE: Need to specify the name of the executable, as
    # ReFrame has no way of knowing that while building from CMake
    executable = './test2'
    # Provide the input through command line arguments
    # e.g., executable_opts = ['arg1', 'arg2'] where arg1, arg2,... are the command
    # line arguments
    executable_opts = ['InputFile']
    builddir ='./build'
    sourcesdir = './src'
    # As a standard convention, we use 4 categories of tags that can help 
    # us run only tests matching certain tag(s). See the top of this file 
    # for the description on types. A user should populate the tagsDict with 
    # the appropriate values for each of the four keys: 'compileOrRun',
    # 'unitOrAggregate', 'cpuOrgpu', 'serialOrParallel'
    tagsDict = {'compileOrRun': 'compile', 'unitOrAggregate':
                'aggregate', 'slowOrFast': 'fast', 'cpuOrgpu': 'cpu',
                'serialOrParallel': 'serial'}
    tags = {x for x in tagsDict.values()}
    # By default ReFrame deletes all the output files generated by the test
    # In case you want to retain any of the files in the output folder of the
    # test, list them in keep_files attribute
    keep_files = ['out_test1']

    @run_before('compile')
    def set_compiler_flags(self):
        # set the make_opts as defined in the Constructor 
        self.build_system.make_opts = self.make_opts
        ''' Set any compilation flag, configuration, environment variables, etc'''
        self.build_system.make_opts = self.make_opts
        #NOTE: In most cases, setting the following attribues within a test should be avoided, 
        #it should be set within the config file. 
        #+ `build_system.cc` string to specify the C compiler
	    #+ `build_system.cflags` list of string to specify the C compiler flags
	    #+ `build_system.cxx` string to specify the C++ compiler
	    #+ `build_system.cxxflags` list of string to specify the C++ compiler flags
	    #+ `build_system.cppflags` list of string to specify the preprocessor flags
	    #+ `build_system.ldflags` list of string to specify the linker flags
	    #+ `build_system.nvcc` string to specify the CUDA compiler
        #+ `variables` a dictionary to set environment variables. e.g, 
        #  self.variables = {'DFT_EFE_LINKER': '"-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"'}
        # Any special configuration that needs to be done externally can be done
        # using the prebuild_cmds attribute 
        # e.g., self.prebuild_cmds = ['./custom_configure -with-mylib']
        # Avoid using it as far as possible, in most cases one can invoke the
        # same configuration through CMake

    @sanity_function
    def validate_test(self):
        '''
        A user-defined function which decides whether the test passed or not as
        well as define an error message to display if the test fails.
        '''
        #NOTE: To make the tests portable, we do not hardcode any values or
        #error message in this file. Everything must be read from the output of
        #the test (either by catching the stdout or by reading from an output
        #file). The error message to display must be specified in this function
        #(or by invoking some other function from here). To integrate with
        #ReFrame's error handling and logging, we pass the boolean storing 
        #whether the test passed or not and the error message to ReFrame's
        # reframe.utility.sanity.assert_true(hasTestPassed, msg) function
        hasTestPassed = True
        msg = 'Failed for some reason'
        bmfilename = "benchmark_test1"
        bmParser = parser.Parser.fromFilename(bmfilename)
        outParser = parser.Parser.fromFilename(evaluate(self.stdout))
        bmVal = bmParser.extractKeyValues("Values")
        outVal = outParser.extractKeyValues("Values")
        hasTestPassed, msg = cu.Compare().cmp(bmVal[0], outVal[0])
        return sn.assert_true(hasTestPassed, msg=msg)

    @run_before('run')
    def set_resources(self):
        # equivalent of --time=hh:mm:ss in slurm
        # The format is a string of the format <days>d<hours>h<minutes>m<seconds>s
        self.time_limit = "0d1h1m1s" 
        # equivalent of --ntasks in slurm
        self.num_tasks = 1
        # equivalent of --ntasks-per-node in slurm
        self.num_tasks_per_node = 1
        # equivalent of --ntasks-per-core in slurm
        self.num_tasks_per_core = 1
        # equivalent of --ntasks-per-socket in slurm
        self.num_tasks_per_socket = 1
        # equivalent of --cpus-per-task in slurm
        self.num_cpus_per_task = 1
        # equivalent of --exclusive in slurm
        self.exclusive_access = True
        # equivalent of '--gres=gpu:{num_gpus_per_node}
        self.num_gpus_per_node = 1


