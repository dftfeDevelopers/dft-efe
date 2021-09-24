import reframe as rfm
import reframe.utility.sanity as sn
import CompareUtil as cu
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
"""


#@rfm.simple_test
#class MakefileTest(rfm.RegressionTest):
#    descr = 'Test demonstrating use of CMake'
#    valid_systems = ['*']
#    valid_prog_environs = ['gnu']
#    #NOTE: Need to specify the name of the executable, as
#    # ReFrame has no way of knowing that while building from CMake
#    executable = './test'
#    build_system = 'CMake'
#    builddir ='./build'
#    sanity_patterns = sn.assert_true(1)
#    
#    @run_before('compile')
#    def set_compiler_flags(self):
#        self.build_system.cxxflags = ['-std=c++11']
#
#    
#    def set_sanity_patterns(self):
#        self.sanity_patterns = sn.assert_found(
#            r'5 5 5 5 5', self.stdout
#        )

@rfm.simple_test
class MakeOnlyTest(rfm.CompileOnlyRegressionTest):
    descr = 'Test demonstrating use of CMake'
    valid_systems = ['*']
    valid_prog_environs = ['*']
    build_system = 'CMake'
    make_opts = ['all']
    builddir ='./build'
    sourcesdir = './src'
    tags = {"compile", "fast"}

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


    @run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_not_found(r'warning/i', self.stdout)


@rfm.simple_test
class RunTestNoExternalInput(rfm.RegressionTest):
    descr = 'Test demonstrating a Regression Test using CMake'
    valid_systems = ['*']
    valid_prog_environs = ['gnu']
    build_system = 'CMake'
    make_opts = ['all']
    #NOTE: Need to specify the name of the executable, as
    # ReFrame has no way of knowing that while building from CMake
    executable = './test'
    builddir ='./build'
    sourcesdir = './src'
    tags={"unit", "fast"}

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
    def validate_test(self):
        '''
        A user-defined function which validates the test
        and returns True if the test passes, False otherwise.
        '''
        #NOTE: To make the tests portable, we do not hardcode any values or
        #error message in this file. Everything must be read from the output of
        #the test (i.e., the computed value, the benchmark value, the error
        #message to print if the test failed must be read from the output of the
        #test. The job of this function is to parse the output and decide
        #whether the test has passed or not. 
        hasTestPassed = True
        msg = 'Failed for some reason'
        return sn.assert_true(hasTestPassed, msg=msg)

@rfm.simple_test
class RunTestWithExternalInput(rfm.RegressionTest):
    descr = 'Test demonstrating a Regression Test using CMake'
    valid_systems = ['*']
    valid_prog_environs = ['gnu']
    build_system = 'CMake'
    make_opts = ['all']
    #NOTE: Need to specify the name of the executable, as
    # ReFrame has no way of knowing that while building from CMake
    executable = './test'
    # Provide the input through command line arguments
    # e.g., executable_opts = ['arg1', 'arg2'] where arg1, arg2,... are the command
    # line arguments
    executable_opts = ['InputFile']
    builddir ='./build'
    sourcesdir = './src'
    tags = {"aggregate", "slow"}

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
    def validate_test(self):
        '''
        A user-defined function which validates the test
        and returns True if the test passes, False otherwise.
        '''
        #NOTE: To make the tests portable, we do not hardcode any values or
        #error message in this file. Everything must be read from the output of
        #the test (i.e., the computed value, the benchmark value, the error
        #message to print if the test failed must be read from the output of the
        #test. The job of this function is to parse the output and decide
        #whether the test has passed or not. 
        hasTestPassed = True
        msg = 'Failed for some reason'
        bmfilename = "benchmark_test1"
        outfilename = "benchmark_test1"
        bmParser = cu.Parser(bmfilename)
        outParser = cu.Parser(outfilename)
        bmVal = bmParser.extractKeyValue("Values")
        outVal = outParser.extractKeyValue("Values")
        hasTestPassed, msg = cu.Compare().cmp(bmVal, outVal)
        return sn.assert_true(hasTestPassed, msg=msg)
