# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import reframe as rfm
import reframe.utility.sanity as sn


@rfm.simple_test
class MakefileTest(rfm.RegressionTest):
    descr = 'Test demonstrating use of CMake'
    valid_systems = ['*']
    valid_prog_environs = ['gnu']
    executable = './test'
    build_system = 'CMake'
    builddir ='./build'

    @run_before('compile')
    def set_compiler_flags(self):
        self.build_system.cxxflags = ['-std=c++11']

    @run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_found(
            r'5 5 5 5 5', self.stdout
        )


@rfm.simple_test
class MakeOnlyTest(rfm.CompileOnlyRegressionTest):
    descr = 'Test demonstrating use of CMake'
    valid_systems = ['*']
    valid_prog_environs = ['gnu']
    build_system = 'CMake'
    builddir ='./build'
    #variables = {'DFT_EFE_LINKER': '"-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"'}

    @run_before('compile')
    def set_compiler_flags(self):
        self.build_system.cxxflags = ['-std=c++11']

    @run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_not_found(r'warning', self.stdout)