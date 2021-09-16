# Instruction to using ReFrame
This is an instruction for developers to use ReFrame HPC testing framework.

## Installation
`module load python/3.9.1`  
`pip install reframe-hpc`

## Getting Start
### File description
`/config`: The environmental settings  
`/src`: This directory contains  toy test to see if daxpy works. There should be a CMakeLists.txt and the cpp file 
test.cpp (The file where contains the main function for the test.)  
`test.py`: The python file which starts the test.

### Example run
Under `${PROJECT_HOME}/test`, run `reframe -C ./config/mysettings.py -c ./test.py -r`. An example of a successful 
output looks like this.  

````
[ReFrame Setup]
  version:           3.8.1
  command:           '${HOME}/.local/bin/reframe -C ./config/mysettings.py -c ./test.py -r'
  launched by:       $USER@gl-login2.arc-ts.umich.edu
  working directory: '${PROJECT_HOME}/test'
  settings file:     './config/mysettings.py'
  check search path: '${PROJECT_HOME}/test/test.py'
  stage directory:   '${PROJECT_HOME}/test/stage'
  output directory:  '${PROJECT_HOME}/test/output'

[==========] Running 2 check(s)
[==========] Started on Thu Sep 16 18:11:53 2021

[----------] started processing MakefileTest (Test demonstrating use of CMake)
[ RUN      ] MakefileTest on greatlakes:login using gnu
[ RUN      ] MakefileTest on greatlakes:compute using gnu
[----------] finished processing MakefileTest (Test demonstrating use of CMake)

[----------] started processing MakeOnlyTest (Test demonstrating use of CMake)
[ RUN      ] MakeOnlyTest on greatlakes:login using gnu
[ RUN      ] MakeOnlyTest on greatlakes:compute using gnu
[----------] finished processing MakeOnlyTest (Test demonstrating use of CMake)

[----------] waiting for spawned checks to finish
[       OK ] (1/4) MakefileTest on greatlakes:compute using gnu [compile: 3.207s run: 6.893s total: 10.135s]
[       OK ] (2/4) MakefileTest on greatlakes:login using gnu [compile: 3.072s run: 10.190s total: 13.303s]
[       OK ] (3/4) MakeOnlyTest on greatlakes:compute using gnu [compile: 3.557s run: 0.590s total: 4.176s]
[       OK ] (4/4) MakeOnlyTest on greatlakes:login using gnu [compile: 3.051s run: 4.212s total: 7.291s]
[----------] all spawned checks have finished

[  PASSED  ] Ran 4/4 test case(s) from 2 check(s) (0 failure(s), 0 skipped)
[==========] Finished on Thu Sep 16 18:12:07 2021
Run report saved in '${HOME}/.reframe/reports/run-report.json'
Log file(s) saved in '/tmp/rfm-h88nvn63.log'
````