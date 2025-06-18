#  Copyright (c) 2022.
#  The Regents of the University of Michigan and DFT-EFE developers.
#
#  This file is part of the DFT-EFE code.
#
#  DFT-EFE is free software: you can redistribute it and/or modify
#    it under the terms of the Lesser GNU General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#  DFT-EFE is distributed in the hope that it will be useful, but
#    WITHOUT ANY WARRANTY; without even the implied warranty
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#    See the Lesser GNU General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public
#    License at the top level of DFT-EFE distribution.  If not, see
#    <https://www.gnu.org/licenses/>.

# @author Ian C. Lin

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

'''
TestUtilsBuild
This test compiles all tests in the DFTEFE_PATH/test/utils folder and copies 
the executable to DFTEFE_PATH/test/utils/executable. 
This test should be run before any RunOnlyTests. 
The BuildOnly test takes CMake options from CMakeConfigOptions.txt stored in
the DFT-EFE main directory to ensure that the CMake configuration is consistent 
across the main executable and the tests. 
The process of building and running a test is as follows

1. set DFTEFE_PATH to the main directory (i.e., "export DFTEFE_PATH="/path/to/dft-efe/")
2. python configure.py in the main directory of DFT-EFE to generate CMakeConfigOptions.txt
3. In the test/atoms directory, run "reframe -C ../config/mysettings.py -c TestAtomsBuild.py -r" 
   to compile the tests.
4. to run all cpu tests, use the command
   reframe -C ../config/mysettings.py -c ./ -R -n 'RunOnlyTest*' -t cpu -r
   to run all gpu tests, use the command (GPU options need to be turned on in CMakeConfigOptions.txt)
   reframe -C ../config/mysettings.py -c ./ -R -n 'RunOnlyTest*' -t gpu -r
'''
@rfm.simple_test
class BuildOnlyTestAtoms(rfm.CompileOnlyRegressionTest):
    descr = 'A build only test using CMake'
    build_system = 'CMake'
    make_opts = ['TestAtomSphericalDataPSP']
    tagsDict = {'compileOrRun': 'compile', 'unitOrAggregate':
                'unit', 'slowOrFast': 'fast', 'arch': 'cpu',
                'serialOrParallel': 'serial'}
    tags = {x.lower() for x in tagsDict.values()}
    valid_systems = ss.getValidSystems(tagsDict['arch']) 
    valid_prog_environs = ['*']
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
            if hasTestPassed:
                bincpy.BinCpy(os.path.dirname(os.path.abspath(__file__)))
            return sn.assert_true(hasTestPassed, msg=msg)


@rfm.simple_test
class BuildAndRunTestAtoms(rfm.RegressionTest):
    target_name = 'TestAtomSphericalDataPSP'
    descr = '''A build and run test for AtomSphericalData.cpp'''
    build_system = 'CMake'
    make_opts = [target_name]
    # NOTE: Need to specify the name of the executable, as
    # ReFrame has no way of knowing that while building from CMake
    executable = "./"+target_name
    tagsDict = {'compileOrRun': 'compile', 'unitOrAggregate':
        'unit','slowOrFast': 'fast', 'arch': 'cpu',
                'serialOrParallel': 'serial'}
    tags = {x.lower() for x in tagsDict.values()}
    valid_systems = ss.getValidSystems(tagsDict['arch']) 
    valid_prog_environs = ['*']
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
                                                   num_tasks_per_node = 2,
                                                   ntasks = 2,
                                                   mem_per_cpu = '2gb')


    @sanity_function
    def validate_test(self):
        # This test does not generate any output. It throws an exception
        # if the logic of MPICommunicatorP2P fails to find the ghost data
        hasAssertFail = True
        hasThrownException = True
        hasError = True
        msgError = '''Found error(s) in
        BuildAndRunTestAtomSphericalDataPSP.'''
        msgThrownException = '''Found exceptions in 
        BuildAndRunTestAtomSphericalDataPSP.'''
        msgAssertFail = '''Found assert fail(s) in
        BuildAndRunTestAtomSphericalDataPSP.'''
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