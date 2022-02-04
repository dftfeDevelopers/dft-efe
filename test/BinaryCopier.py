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

import os
import glob
import shutil


'''
This is an auxiliary file for copying binaries from the ReFrame staged folder to an executable folder created externally
for RunOnlyTest. To use this copier, this BinaryCopier.py has to be imported, i.e. the following line has to be added to
the beginning of the file
    bincpy = rfm.utility.import_module_from_file(DFTEFE_PATH+"/test/BinaryCopier.py")
, and the following line has to be added to the test python file.
    bincpy.BinCpy(os.path.dirname(os.path.abspath(__file__)))
'''

'''
Returns true if a file is an executable.
It excludes shell scripts (i.e., those ending with .sh).
'''
def isFileExecutable(fpath):
    return (os.access(fpath, os.X_OK))  and (fpath[-3:] != '.sh')


def BinCpy(dir):
    [dir_path, dir_name] = os.path.split(dir)
    exe_path = dir_path+"/"+dir_name+"/executable"
    if not os.path.isdir(exe_path):
        os.mkdir(exe_path)

    for filename in os.listdir(os.getcwd()):
        f = os.path.join(os.getcwd(), filename)
        # checking if it is a file
        if os.path.isfile(f):
            # checking it is an executable 
            if isFileExecutable(f):
                shutil.copy2(f, exe_path)
