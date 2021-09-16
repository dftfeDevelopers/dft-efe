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
Under `$PROJECT_HOME/test`, run `reframe -C ./config/mysettings.py -c ./test.py -r`. An example output looks like 
this.  

````
[ReFrame Setup]  
version:           3.8.1  
command:           'reframe -C ./config/mysettings.py -c ./test.py -r'  
launched by:       $USER@gl-login2.arc-ts.umich.edu  
working directory: '$PROJECT_HOME/test'  
settings file:     './config/mysettings.py'  
check search path: '$PROJECT_HOME/test/test.py'  
stage directory:   '$PROJECT_HOME/test/stage'  
output directory:  '$PROJECT_HOME/test/output'

[==========] Running 0 check(s)  
[==========] Started on Thu Sep 16 16:03:57 2021

[----------] waiting for spawned checks to finish  
[----------] all spawned checks have finished

[  PASSED  ] Ran 0/0 test case(s) from 0 check(s) (0 failure(s), 0 skipped)  
[==========] Finished on Thu Sep 16 16:03:57 2021  
Run report saved in '$HOME/.reframe/reports/run-report.json'  
Log file(s) saved in '/tmp/rfm-u5fn74m6.log'
````