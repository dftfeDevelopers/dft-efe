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
cpu: Tests to be run on both gpu and cpu
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
    make_opts = []

    tagsDict = {'compileOrRun': 'compile', 'unitOrAggregate':
                'unit', 'slowOrFast': 'fast', 'arch': 'gpu',
                'serialOrParallel': 'serial'}
    tags = {x.lower() for x in tagsDict.values()}

    valid_systems = ['greatlakes:login']
    valid_prog_environs = ['builtin']
    config_opts = cmflags.getConfig(tagsDict['arch'])

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
            if hasTestPassed:
                bincpy.BinCpy(os.path.dirname(os.path.abspath(__file__)))
            return sn.assert_true(hasTestPassed, msg=msg)