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
TestLinearAlgebraBuild
This test compiles all tests and copy the executable to /path/to/test/folder/executable. This test should be run before
any RunOnlyTests. The BuildOnly test takes CMake options from CMakeConfigOptions.txt stored in the DFT-EFE main 
directory to ensure the CMake configuration is consistent for main executable and the tests. The process of building and
running a test is as follows

1. run configure.py in the main directory of DFT-EFE to generate CMakeConfigOptions.txt
2. export DFTEFE_PATH as an environmental variable by "export DFTEFE_PATH="/path/to/dft-efe/"
3. in the test/linearAlgebra directory, run "reframe -C ../config/mysettings.py -c TestLinearAlgebraBuild.py -r" to 
   compile the tests.
4. to run all cpu tests, use the command
   reframe -C ../config/mysettings.py -c ./ -R -n 'RunOnlyTest_*' -t cpu -r
   to run all gpu tests, use the command (GPU options need to be turned on in CMakeConfigOptions.txt)
   reframe -C ../config/mysettings.py -c ./ -R -n 'RunOnlyTest_*' -t gpu -r
'''
@rfm.simple_test
class BuildOnly_TestLinearAlgebra(rfm.CompileOnlyRegressionTest):
    descr = 'A build only test using CMake'
    build_system = 'CMake'
    make_opts = []

    tagsDict = {'compileOrRun': 'compile'}
    tags = {x.lower() for x in tagsDict.values()}

    valid_systems = ['greatlakes:login']
    valid_prog_environs = ['builtin']
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
            if hasTestPassed:
                bincpy.BinCpy(os.path.dirname(os.path.abspath(__file__)))
            return sn.assert_true(hasTestPassed, msg=msg)