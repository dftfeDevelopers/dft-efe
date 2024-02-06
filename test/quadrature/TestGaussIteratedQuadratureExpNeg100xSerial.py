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

@rfm.simple_test
class TestGaussIteratedQuadratureExpNeg100xSerialBuildOnly(rfm.CompileOnlyRegressionTest):
    descr = '''Compile only test for Gauss Iterated quadrature rule in a unit cube'''\
            '''i using exp(-100x) as test function'''
    build_system = 'CMake'
    make_opts = ['TestGaussIteratedQuadratureExpNeg100xSerial']
    sourcesdir = './src'
    # As a standard convention, we use 4 categories of tags that can help 
    # us run only tests matching certain tag(s). See the top of this file 
    # for the description on types. A user should populate the tagsDict with 
    # the appropriate values for each of the four keys: 'compileOrRun',
    # 'unitOrAggregate', 'arch', 'serialOrParallel'
    tagsDict = {'compileOrRun': 'compile', 'unitOrAggregate':
                'aggregate', 'slowOrFast': 'fast', 'arch': 'cpu',
                'serialOrParallel': 'serial'}
    tags = {x.lower() for x in tagsDict.values()}
    valid_systems = ss.getValidSystems(tagsDict['arch']) 
    valid_prog_environs = ['*']
    keep_files = []
    config_opts = cmflags.getConfig()

    @run_before('compile')
    def set_compiler_flags(self):
        self.build_system.make_opts = self.make_opts
        self.build_system.config_opts = self.config_opts

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
class TestGaussIteratedQuadratureExpNeg100xSerialBuildAndRun(rfm.RegressionTest):
    descr = '''Compile and run test for Gauss Iterated quadrature rule in a unit cube'''\
            ''' using exp(-100x) as test function'''
    build_system = 'CMake'
    make_opts = ['TestGaussIteratedQuadratureExpNeg100xSerial']
    executable = './TestGaussIteratedQuadratureExpNeg100xSerial'
    sourcesdir = './src'
    tagsDict = {'compileOrRun': 'compile', 'unitOrAggregate':
                'aggregate', 'slowOrFast': 'fast', 'arch': 'cpu',
                'serialOrParallel': 'serial'}
    tags = {x.lower() for x in tagsDict.values()}
    valid_systems = ss.getValidSystems(tagsDict['arch']) 
    valid_prog_environs = ['*']
    keep_files = []
    config_opts = cmflags.getConfig()

    @run_before('compile')
    def set_compiler_flags(self):
        self.build_system.make_opts = self.make_opts
        self.build_system.config_opts = self.config_opts
    
    @run_before('run')
    def set_launcher_and_resources(self):
        if "serial" in self.tags:
            self.job.launcher = getlauncher('local')()

        if "parallel" in self.tags:
            self.job.launcher.options = ['-n 2']
            self.extra_resources = ss.setResources(self.tagsDict['arch'], 
                                                   time_limit = "00:05:00", 
                                                   num_nodes = 1, 
                                                   num_tasks_per_node = 8,
                                                   ntasks = 8,
                                                   mem_per_cpu = '2gb')

    @sanity_function
    def validate_test(self):
        # This test generates output to a data file. 
        hasAssertFail = True
        hasThrownException = True
        hasError = True
        msgError = '''Found error(s) in
        Running.'''
        msgThrownException = '''Found exceptions in 
        Running.'''
        msgAssertFail = '''Found assert fail(s) in
        Running.'''
        matchesOut = evaluate(sn.findall(r'(?i)error', evaluate(self.stdout)))
        matchesErr = evaluate(sn.findall(r'(?i)error', evaluate(self.stderr)))
        if len(matchesOut) == 0 and len(matchesErr) == 0:
            hasError = False

        matchesOut = evaluate(sn.findall(r'(?i)assert', evaluate(self.stdout)))
        matchesErr = evaluate(sn.findall(r'(?i)assert', evaluate(self.stderr)))
        if len(matchesOut) == 0 and len(matchesErr) == 0:
            hasAssertFail = False
        
        matchesOut = evaluate(sn.findall(r'(?i)throw', evaluate(self.stdout)))
        matchesErr = evaluate(sn.findall(r'(?i)throw', evaluate(self.stderr)))
        if len(matchesOut) == 0 and len(matchesErr) == 0:
            hasThrownException = False
        
        hasTestPassed = not any([hasError, hasAssertFail, hasThrownException]) 
        
        msg = ""
        if hasError:
            msg = msgError

        elif hasAssertFail:
            msg = msgAssertFail

        elif hasThrownException:
            msg = msgThrownException

        else:
            msg=""

        return sn.assert_true(hasTestPassed, msg=msg)
