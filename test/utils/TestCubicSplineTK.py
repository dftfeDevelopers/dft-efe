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
class BuildOnlyTestCubicSplineTK(rfm.CompileOnlyRegressionTest):
    descr = '''Compile only test for cubic spline using TK'''
    build_system = 'CMake'
    make_opts = ['TestCubicSplineTK']
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

    @sanity_function
    def validate_test(self):
        hasWarning = True
        hasError = True
        msgWarning = "Found warning(s) while compiling."
        msgError = "Found error(s) while compiling."
        matches = evaluate(sn.findall(r'(?i)warning', evaluate(self.stdout)))
        if len(matches) == 0:
            hasWarning = False

        matchesOut = evaluate(sn.findall(r'(?i)error', evaluate(self.stdout)))
        matchesErr = evaluate(sn.findall(r'(?i)error', evaluate(self.stderr)))
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
