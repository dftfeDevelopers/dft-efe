#  Copyright (c) 2021.
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

# import CompareUtil as cu
parser = rfm.utility.import_module_from_file("../Parser.py")
cu = rfm.utility.import_module_from_file("../CompareUtil.py")
ss = rfm.utility.import_module_from_file("../SetupSystems.py")
cmflags = rfm.utility.import_module_from_file("../CMakeFlagsParser.py")


@rfm.simple_test
class MultipleExeTest(rfm.RegressionTest):
    descr = 'Dummy Test for showing how to check multiple executables in one test'
    valid_systems = ['greatlakes:login']
    valid_prog_environs = ['builtin']
    build_system = 'CMake'
    make_opts = ['TestVectorAggregate1', 'TestVectorAggregate2', 'TestVectorNorms']
    cmflags.getConfig()
    config_opts = cmflags.getConfig('greatlakes_cpu')
    executable = 'date'
    builddir = './TestVectorAggregate'
    sourcesdir = './src'
    tagsDict = {'compileOrRun': 'Run', 'unitOrAggregate':
        'Aggregate', 'slowOrFast': 'fast', 'arch': 'gpu',
                'serialOrParallel': 'serial'}

    tags = {x for x in tagsDict.values()}

    @run_before('compile')
    def set_compiler_flags(self):
        # set the make_opts as defined in the Constructor 
        self.build_system.make_opts = self.make_opts
        self.build_system.config_opts = self.config_opts

    # @run_before('run')
    # def set_launcher_and_resources(self):
    #     if "serial" in self.tags:
    #         self.job.launcher = getlauncher('local')()
    #
    #     if "parallel" in self.tags:
    #         self.job.launcher.options = ['']
    #     self.extra_resources = ss.setResources(self.tagsDict['arch'])

    @run_before('run')
    def pre_launch(self):
        exe_names = ['./TestVectorAggregate1', './TestVectorAggregate2', './TestVectorNorms']
        cmd = self.job.launcher.run_command(self.job)
        self.prerun_cmds = [
            # f'{cmd} -n {1} {exe}'
            f'{exe}'
            for exe in exe_names
        ]

    @sanity_function
    def validate_test(self):
        hasTestPassed = True
        msg = 'Passed'
        bmfilename = ['TestVectorAggregate1', 'TestVectorAggregate2']
        for filename in bmfilename:
            bmParser = parser.Parser.fromFilename(filename+'.benchmark')
            outParser = parser.Parser.fromFilename(filename+'.out')
            testSet = ["double add", r"double \+=", "double -=", "complex<double> add", "complex<double> \+=",
                       "complex<double> -="]
            for testString in testSet:
                bmVal = bmParser.extractKeyValues(testString)
                outVal = outParser.extractKeyValues(testString)
                hasTestPassed, norm, msg = cu.Compare().cmp(bmVal, outVal, 1.0e-16, 'absolute', 'point')
                if not hasTestPassed:
                    print(filename)
                    print(testString)
                    print('benchmark')
                    print(bmVal)
                    print('output')
                    print(outVal)
                    msg = "Failed in {}".format(testString)
                    return sn.assert_true(hasTestPassed, msg=msg)

        bmfilename = ['TestVectorNorms']
        for filename in bmfilename:
            bmParser = parser.Parser.fromFilename(filename+'.benchmark')
            outParser = parser.Parser.fromFilename(filename+'.out')
            testSet = ["double vec norms", "complex double vec norms"]
            for testString in testSet:
                bmVal = bmParser.extractKeyValues(testString)
                outVal = outParser.extractKeyValues(testString)
                hasTestPassed, norm, msg = cu.Compare().cmp(bmVal, outVal, 1.0e-16, 'absolute', 'point')
                if not hasTestPassed:
                    print(filename)
                    print(testString)
                    print('benchmark')
                    print(bmVal)
                    print('output')
                    print(outVal)
                    msg = "Failed in {}".format(testString)
                    return sn.assert_true(hasTestPassed, msg=msg)

        return sn.assert_true(hasTestPassed, msg=msg)
