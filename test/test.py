import reframe as rfm
import reframe.utility.sanity as sn
from reframe.utility.sanity import evaluate
from reframe.core.backends import getlauncher
#import CompareUtil as cu
parser = rfm.utility.import_module_from_file("parser.py")
cu = rfm.utility.import_module_from_file("compareUtil.py")
ss = rfm.utility.import_module_from_file("setupSystems.py")
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

@rfm.simple_test
class BuildOnlyAll(rfm.CompileOnlyRegressionTest):
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
    # 'unitOrAggregate', 'arch', 'serialOrParallel'
    tagsDict = {'compileOrRun': 'compile', 'unitOrAggregate':
                'unit', 'slowOrFast': 'fast', 'arch': 'cpu',
                'serialOrParallel': 'serial'}
    tags = {x.lower() for x in tagsDict.values()}

    #Define valid_systems as a list of strings of the format
    #'system:partiion'. Over here, based on the 'arch' 
    #(i.e,,'cpu', 'gpu', or 'both'), we filter out
    #the system:partitions from a list named system_partition_list
    #in setupSystems.py. The convention is for 'arch'='cpu', we
    #select only those system:partitions that do not contain the string 'gpu' 
    #in it. Conversely, for 'arch'='gpu', only those system:partitions are
    #included which contains the string 'gpu'. For 'arch'='both', all the
    #system:partiions are selected.
    #NOTE: For any new systems:partition added to the config file, 
    # they must also be added to the system_partition_in setupSystems.py
    valid_systems = ss.getValidSystems(tagsDict['arch']) 
    valid_prog_environs = ['*']


    @run_before('compile')
    def set_compiler_flags(self):
        self.build_system.make_opts = self.make_opts
        ''' Set any compilation flag, configuration, environment variables, etc'''
        #NOTE: In most cases, setting the following attribues within a test should be avoided, 
        #it should be set within the config file. 
        #+ `self.build_system.cc` string to specify the C compiler
	    #+ `self.build_system.cflags` list of string to specify the C compiler flags
	    #+ `self.build_system.cxx` string to specify the C++ compiler
	    #+ `self.build_system.cxxflags` list of string to specify the C++ compiler flags
	    #+ `self.build_system.cppflags` list of string to specify the preprocessor flags
	    #+ `self.build_system.ldflags` list of string to specify the linker flags
	    #+ `self.build_system.nvcc` string to specify the CUDA compiler
        #+ `self.variables` a dictionary to set environment variables. e.g, 
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
        msg = ""
        if hasError:
            msg = msgError

        elif hasWarning:
            msg = msgWarning

        else:
            msg = ""
            return sn.assert_true(hasTestPassed, msg=msg)




@rfm.simple_test
class FileOutRunTest(rfm.RegressionTest):
    descr = 'Regression Test using CMake and output to a file'
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
    # 'unitOrAggregate', 'arch', 'serialOrParallel'
    tagsDict = {'compileOrRun': 'compile', 'unitOrAggregate':
                'unit','slowOrFast': 'fast', 'arch': 'cpu',
                'serialOrParallel': 'serial'}
    tags = {x.lower() for x in tagsDict.values()}

    #Define valid_systems as a list of strings of the format
    #'system:partiion'. Over here, based on the 'arch' 
    #(i.e,,'cpu', 'gpu', or 'both'), we filter out
    #the system:partitions from in setupSystems.py. 
    #The convention is for 'arch'='cpu', we
    #select only those system:partitions that do not contain the string 'gpu' 
    #in it. Conversely, for 'arch'='gpu', only those system:partitions are
    #included which contains the string 'gpu'. For 'arch'='both', all the
    #system:partiions are selected.
    valid_systems = ss.getValidSystems(tagsDict['arch']) 
    valid_prog_environs = ['*']
    
    # By default ReFrame deletes all the output files generated by the test
    # In case you want to retain any of the files in the output folder of the
    # test, list them in keep_files attribute
    keep_files = ['out_test1']

    @run_before('compile')
    def set_compiler_flags(self):
        ''' Set any compilation flag, configuration, environment variables, etc'''
        # set the make_opts as defined in the Constructor 
        self.build_system.make_opts = self.make_opts
        #NOTE: In most cases, setting the following attribues within a test should be avoided, 
        #it should be set within the config file. 
        #+ `self.build_system.cc` string to specify the C compiler
	    #+ `self.build_system.cflags` list of string to specify the C compiler flags
	    #+ `self.build_system.cxx` string to specify the C++ compiler
	    #+ `self.build_system.cxxflags` list of string to specify the C++ compiler flags
	    #+ `self.build_system.cppflags` list of string to specify the preprocessor flags
	    #+ `self.build_system.ldflags` list of string to specify the linker flags
	    #+ `self.build_system.nvcc` string to specify the CUDA compiler
        #+ `self.variables` a dictionary to set environment variables. e.g, 
        #  self.variables = {'DFT_EFE_LINKER': '"-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"'}
        # Any special configuration that needs to be done externally can be done
        # using the prebuild_cmds attribute 
        # e.g., self.prebuild_cmds = ['./custom_configure -with-mylib']
        # Avoid using it as far as possible, in most cases one can invoke the
        # same configuration through CMake
    
    @run_before('run')
    def set_launcher_and_resources(self):
        #By default in the config file the launcher for most systems:partitions
        #is set to srun or mpirun. However, for a serial job, we want to set it
        #to the local launcher (i.e., just ./exec)
        if "serial" in self.tags:
            self.job.launcher = getlauncher('local')()
        
        if "parallel" in self.tags:
            #For jobs launched through srun or mpirun, use the following
            #to set any launcher options
            #Example1: For srun -n 2 ./exec,
            # set the options as self.job.launcher.options = ['-n 2]
            #Example2: For binding cpu to cores 
            #self.job.launcher.options = ['--cpu-bind=cores']
            #self.job.launcher.options = ['launcher options to set']
            #By default it is kept blank (as defined below)
            self.job.launcher.options = ['']
   
        #In order to simplify the resource allocation procedure in a way that
        #is agnostic of the queueing system (Slurm, PBS, Torque, etc.), we use
        #the setResources() function defined in setupSystems.py. The
        #setResources() takes in the following paramters:
        #1. archTag: string that can be 'cpu', 'gpu', or 'both' (Default: 'both')
        #2. time_limit: string of the format "hrs:mins:secs" 
        #               (Default: "00:02:00")
        #3. num_nodes: integer for number of nodes to allocate (Default: 1)
        #4. num_tasks_per_node: integer for number of tasks to use per node 
        #                        (Default: 1)
        #5. mem_per_cpu: string of the format "<number>mb" or "<number>gb" for
        #                 the memory to allocate per cpu (Default: "2gb")
        #6. gpus_per_node: integer for number of gpus to allocate per node.
        #                  This is used only when the archTag='cpu' or 
        #                   archTag='both'. (Default: 1) 
        #Internally, setResources() function instantiates the 'resources' 
        #related placeholders defined in the cofig file. See REFRAME.md
        #for more details
        self.extra_resources = ss.setResources(self.tagsDict['arch'])

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
        hasTestPassed, norm, msg = cu.Compare().cmp(bmVal[0], outVal[0],
                                                    normType='inf')
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
    # 'unitOrAggregate', 'arch', 'serialOrParallel'
    tagsDict = {'compileOrRun': 'compile', 'unitOrAggregate':
                'aggregate', 'slowOrFast': 'fast', 'arch': 'cpu',
                'serialOrParallel': 'serial'}
    tags = {x for x in tagsDict.values()}
    
    #Define valid_systems as a list of strings of the format
    #'system:partiion'. Over here, based on the 'arch' 
    #(i.e,,'cpu', 'gpu', or 'both'), we filter out
    #the system:partitions from a list named system_partition_list
    #in setupSystems.py. The convention is for 'arch'='cpu', we
    #select only those system:partitions that do not contain the string 'gpu' 
    #in it. Conversely, for 'arch'='gpu', only those system:partitions are
    #included which contains the string 'gpu'. For 'arch'='both', all the
    #system:partiions are selected.
    #NOTE: For any new systems:partition added to the config file, 
    # they must also be added to the system_partition_in setupSystems.py
    valid_systems = ss.getValidSystems(tagsDict['arch']) 
    valid_prog_environs = ['*']


    # By default ReFrame deletes all the output files generated by the test
    # In case you want to retain any of the files in the output folder of the
    # test, list them in keep_files attribute
    keep_files = ['out_test1']

    @run_before('compile')
    def set_compiler_flags(self):
        # set the make_opts as defined in the Constructor 
        self.build_system.make_opts = self.make_opts
        ''' Set any compilation flag, configuration, environment variables, etc'''
        #NOTE: In most cases, setting the following attribues within a test should be avoided, 
        #it should be set within the config file. 
        #+ `self.build_system.cc` string to specify the C compiler
	    #+ `self.build_system.cflags` list of string to specify the C compiler flags
	    #+ `self.build_system.cxx` string to specify the C++ compiler
	    #+ `self.build_system.cxxflags` list of string to specify the C++ compiler flags
	    #+ `self.build_system.cppflags` list of string to specify the preprocessor flags
	    #+ `self.build_system.ldflags` list of string to specify the linker flags
	    #+ `self.build_system.nvcc` string to specify the CUDA compiler
        #+ `self.variables` a dictionary to set environment variables. e.g, 
        #  self.variables = {'DFT_EFE_LINKER': '"-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"'}
        # Any special configuration that needs to be done externally can be done
        # using the prebuild_cmds attribute 
        # e.g., self.prebuild_cmds = ['./custom_configure -with-mylib']
        # Avoid using it as far as possible, in most cases one can invoke the
        # same configuration through CMake
    
    @run_before('run')
    def set_launcher_and_resources(self):
        #By default in the config file the launcher for most systems:partitions
        #is set to srun or mpirun. However, for a serial job, we want to set it
        #to the local launcher (i.e., just ./exec)
        if "serial" in self.tags:
            self.job.launcher = getlauncher('local')()
        
        if "parallel" in self.tags:
            #For jobs launched through srun or mpirun, use the following
            #to set any launcher options
            #Example1: For srun -n 2 ./exec,
            # set the options as self.job.launcher.options = ['-n 2]
            #Example2: For binding cpu to cores 
            #self.job.launcher.options = ['--cpu-bind=cores']
            #self.job.launcher.options = ['launcher options to set']
            #By default it is kept blank (as defined below)
            self.job.launcher.options = ['']
   
        #In order to simplify the resource allocation procedure in a way that
        #is agnostic of the queueing system (Slurm, PBS, Torque, etc.), we use
        #the setResources() function defined in setupSystems.py. The
        #setResources() takes in the following paramters:
        #1. archTag: string that can be 'cpu', 'gpu', or 'both' (Default: 'both')
        #2. time_limit: string of the format "hrs:mins:secs" 
        #               (Default: "00:02:00")
        #3. num_nodes: integer for number of nodes to allocate (Default: 1)
        #4. num_tasks_per_node: integer for number of tasks to use per node 
        #                        (Default: 1)
        #5. mem_per_cpu: string of the format "<number>mb" or "<number>gb" for
        #                 the memory to allocate per cpu (Default: "2gb")
        #6. gpus_per_node: integer for number of gpus to allocate per node.
        #                  This is used only when the archTag='cpu' or 
        #                   archTag='both'. (Default: 1) 
        #Internally, setResources() function instantiates the 'resources' 
        #related placeholders defined in the cofig file. See REFRAME.md
        #for more details
        self.extra_resources = ss.setResources(self.tagsDict['arch'])

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
        hasTestPassed, norm, msg = cu.Compare().cmp(bmVal[0], outVal[0])
        return sn.assert_true(hasTestPassed, msg=msg)


@rfm.simple_test
class ParameterizedNumProcsTest(rfm.RegressionTest):
    descr = '''Regression Test using CMake and stdout that parameterizes the
    number of processors'''
    #Register the list of values a parameter takes as a ReFrame parameter
    #datatype. Without registering as a parameter datatype, ReFrame has no
    #way of generating different test for each value of the parameter
    procs_list = parameter([1,2,4])
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
    # 'unitOrAggregate', 'arch', 'serialOrParallel'
    tagsDict = {'compileOrRun': 'compile', 'unitOrAggregate':
                'aggregate', 'slowOrFast': 'fast', 'arch': 'cpu',
                'serialOrParallel': 'serial'}
    tags = {x for x in tagsDict.values()}
    
    #Define valid_systems as a list of strings of the format
    #'system:partiion'. Over here, based on the 'arch' 
    #(i.e,,'cpu', 'gpu', or 'both'), we filter out
    #the system:partitions from a list named system_partition_list
    #in setupSystems.py. The convention is for 'arch'='cpu', we
    #select only those system:partitions that do not contain the string 'gpu' 
    #in it. Conversely, for 'arch'='gpu', only those system:partitions are
    #included which contains the string 'gpu'. For 'arch'='both', all the
    #system:partiions are selected.
    #NOTE: For any new systems:partition added to the config file, 
    # they must also be added to the system_partition_in setupSystems.py
    valid_systems = ss.getValidSystems(tagsDict['arch']) 
    valid_prog_environs = ['*']


    # By default ReFrame deletes all the output files generated by the test
    # In case you want to retain any of the files in the output folder of the
    # test, list them in keep_files attribute
    keep_files = ['out_test1']

    @run_before('compile')
    def set_compiler_flags(self):
        # set the make_opts as defined in the Constructor 
        self.build_system.make_opts = self.make_opts
        ''' Set any compilation flag, configuration, environment variables, etc'''
        #NOTE: In most cases, setting the following attribues within a test should be avoided, 
        #it should be set within the config file. 
        #+ `self.build_system.cc` string to specify the C compiler
	    #+ `self.build_system.cflags` list of string to specify the C compiler flags
	    #+ `self.build_system.cxx` string to specify the C++ compiler
	    #+ `self.build_system.cxxflags` list of string to specify the C++ compiler flags
	    #+ `self.build_system.cppflags` list of string to specify the preprocessor flags
	    #+ `self.build_system.ldflags` list of string to specify the linker flags
	    #+ `self.build_system.nvcc` string to specify the CUDA compiler
        #+ `self.variables` a dictionary to set environment variables. e.g, 
        #  self.variables = {'DFT_EFE_LINKER': '"-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"'}
        # Any special configuration that needs to be done externally can be done
        # using the prebuild_cmds attribute 
        # e.g., self.prebuild_cmds = ['./custom_configure -with-mylib']
        # Avoid using it as far as possible, in most cases one can invoke the
        # same configuration through CMake
    
    @run_before('run')
    def set_launcher_and_resources(self):
        #By default in the config file the launcher for most systems:partitions
        #is set to srun or mpirun. However, for a serial job, we want to set it
        #to the local launcher (i.e., just ./exec)
        if "serial" in self.tags:
            self.job.launcher = getlauncher('local')()
        
        if "parallel" in self.tags:
            #For jobs launched through srun or mpirun, use the following
            #to set any launcher options
            #Example1: For srun -n 2 ./exec,
            # set the options as self.job.launcher.options = ['-n 2]
            #Example2: For binding cpu to cores 
            #self.job.launcher.options = ['--cpu-bind=cores']
            #self.job.launcher.options = ['launcher options to set']
            
            
            #Since this example intends to parameterize the number of
            #processors, we modify the launcher option to redefine the 
            #number of processors for each instance of the paramtererized 
            #test
            self.job.launcher.options = [f'-n {self.procs_list}']
   
        #In order to simplify the resource allocation procedure in a way that
        #is agnostic of the queueing system (Slurm, PBS, Torque, etc.), we use
        #the setResources() function defined in setupSystems.py. The
        #setResources() takes in the following paramters:
        #1. archTag: string that can be 'cpu', 'gpu', or 'both' (Default: 'both')
        #2. time_limit: string of the format "hrs:mins:secs" 
        #               (Default: "00:02:00")
        #3. num_nodes: integer for number of nodes to allocate (Default: 1)
        #4. num_tasks_per_node: integer for number of tasks to use per node 
        #                        (Default: 1)
        #5. mem_per_cpu: string of the format "<number>mb" or "<number>gb" for
        #                 the memory to allocate per cpu (Default: "2gb")
        #6. gpus_per_node: integer for number of gpus to allocate per node.
        #                  This is used only when the archTag='cpu' or 
        #                   archTag='both'. (Default: 1) 
        #Internally, setResources() function instantiates the 'resources' 
        #related placeholders defined in the cofig file. See REFRAME.md
        #for more details
        self.extra_resources = ss.setResources(self.tagsDict['arch'])

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
        hasTestPassed, norm, msg = cu.Compare().cmp(bmVal[0], outVal[0])
        return sn.assert_true(hasTestPassed, msg=msg)
