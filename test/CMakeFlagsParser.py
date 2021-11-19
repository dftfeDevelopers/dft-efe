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

import os

def getConfig(flag_type=""):
    dftefe_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cmake_config_opts = []
    if flag_type == 'greatlakes_cpu':
        cmake_config_opts = [
            '-DDFTEFE_BLAS_LIBRARIES="-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"',
            '-DDFTEFE_SCALAPACK_LIBRARIES="-L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -lgomp -lpthread -lm -ldl"',
            '-DCMAKE_PREFIX_PATH="/home/vikramg/DFT-FE-softwares/dealiiDevCustomized/install_gcc8.2.0_openmpi4.0.6_minimal"',
            '-DENABLE_CUDA=OFF']
        cmake_config_opts += ['-DDFTEFE_PATH="'+dftefe_path+'"']
    return cmake_config_opts
