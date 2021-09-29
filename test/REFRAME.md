# Instruction to using ReFrame

## ReFrame Guide
https://reframe-hpc.readthedocs.io/en/stable/

## Installation
+ Download ReFrame to a directory (preferably put it in dftefe/test directory)
  ```
  cd dftefe/test
  git clone https://github.com/eth-cscs/reframe.git
  ```
+ Make sure that you have a python3.7 or higher available in your PATH 
(e.g., `module load python3.7` or `module load python 3.9`)

+ Change directory to ReFrame directory and run the ./bootstrap.sh script
  ```
  cd reframe
  ./bootstrap.sh
  ```
+ Verify the installation by running the following to display the version
  ```
  ./bin/reframe -V
  ```


## ReFrame Regression Test Basics
Although ReFrame provides many features, we will be using a limited number of features. Each test must be a python class. Typically, a test class should be
split into four parts: 
+ A constructor part which sets the important attributes (source dir, build system type, etc)
+ A compiler flag setter to set any compilation flag or environment   
+ A test validation part which decides whether the test passed or failed and define a message to display if the test fails
+ An optional resource allocation part (useful when running in parallel or requesting access via a queueing system)
### Constructor part
The following are the usual attributes that one requires to set in each test class. The ones marked as __required__ 
are the mandatory ones that must be specified for each test class.
+ `valid_systems` list of 'system:partition' on which to run the test (__required__)
   Examples:  
	+ `valid_systems = ['*']` to specify all the system:paritions defined in the config file
	+ `valid_systems = ['greatlakes:*']` to specify all the partitions for the greatlakes system
	+ `valid_systems = ['greatlakes:compute', 'cori:login']` to specify greatlakes compute parition 
	   and cori login parition 
	+ You can provide any regular expression (regex) to filter the system:partition from the config file  
   __Recommended__: `valid_systems = ['*']` 
+ `valid_prog_environs` list of programming environment to run the test in (__required__)
  Examples:
	+ `valid_prog_environs = ['*']` to specify all the programming environments defined in the config file
	+ `valid_prog_environs = ['gnu', 'intel']`  to use both gnu and intel environments, as defined in the config file  
  __Recommended__: `valid_prog_environs = ['*']`
+ `sourcepath` string containing path to source file. Use it when you have a single source file to test (__required__ if `sourcesdir` is not set)
   Examples `source_path = 'test.cc'`
+ `sourcesdir` string containing the path to the source directory. Use it when you have multiple source files (__required__ if `sourcepath` is not set) 
+ `executable_opts` string containing command line arguments to the test executable (optional)
  Examples:
	+ `executable_opts = '> outfile'` to redirect the `stdout` to outfile
        + `executable_opts = 'arg1 arg2'` to provide arg1 and arg2 as command line input 

+ `descr` description of the test (optional)
+ `build_system` string to define the build system (optional). 
   Example: 
	+`build_system = 'CMake'` to indicate CMake based build) 

+ `extra_resources` a dictionary setting resources that should be allocated to the test by the scheduler. 
   Typically, the resources are defined in the configuration of a system partition, e.g,  
  ```
  'resources': 
    [{ 
        'name': 'gpu',
        'options': ['--gres=gpu:{num_gpus_per_node}'] 	
     },
     {
	'name': 'cpu',
        'options': [
            '--mem={memory}',
            '--time={time}'
        ]
     }
   ]
  ```
In that case, a test class then specify the resources using the `extra_resources` attribute, e.g.,
  ```
  self.extra_resources = {
    'gpu': {'num_gpus_per_node': 2}
    'cpu': {
        'memory': '4GB',
        'time': '48:00:00'
    }
  }
  ```
### Compiler setting part 
For each test, one might need to set certain compilation and environment flags. We do this within the following member function
  ```    
  @run_before('compile')
  def set_compiler_flags(self):
  ```
The ```@run_before('compile')``` ReFrame decorator marks the ```set_compiler_flags(self)``` function as something that 
should run prior to any compilation. The following are commonly used attributes that one might need to set in this function
+ `self.build_system.make_opts` list of strings to be passed as command line options to cmake (optional)
  Typically, this should be used to specify any target for cmake.For example, 
  if the CMakeLists contains various targets and we want to test only one of them, 
  we can provide the target here. e.g., `self.build_system.make_opts` = ['all'] 
+ There are several `build_system` attributes (e.e., compiler type, compiler flags, linker flags, etc). 
  One can set these attributes within the ReFrame test class. However, in most circumstances, _one should avoid_, 
  setting these attributes within a ReFrame test class. As far as possible, these attribues should be set globally for
  all tests within the config file. Important `build_system` attributes are:
	+ `self.build_system.cc` string to specify the C compiler
	+ `self.build_system.cflags` list of string to specify the C compiler flags
	+ `self.build_system.cxx` string to specify the C++ compiler
	+ `self.build_system.cxxflags` list of string to specify the C++ compiler flags
	+ `self.build_system.cppflags` list of string to specify the preprocessor flags
	+ `self.build_system.ldflags` list of string to specify the linker flags
	+ `self.build_system.nvcc` string to specify the CUDA compiler


+ `current_system` an object containing info of the the system the regression test is currently executing on
   Usage:
	+ `current_system.descr` a string containing the description of the system (usually provided in the config file)
	+ `current_system.name` a string containing the name of the system
	+ `current_system.paritions` a list of string containing the partitions in the system
	+ More attributes on https://reframe-hpc.readthedocs.io/en/stable/regression\_test\_api.html#reframe.core.systems.System
+ `current_partition` an object containing info of the system partition the regression test is currently executing on (usually provided in the config file)
  Usage:
	+ `curren_partition.name` a string containing the name of the partition
	+ `current_partition.descr` a string containing the description of the parition
	+ `current_partition.access` list of string containing scheduler options for accessing this system partition 
	+ `current_partition.environs` list of string containing the programming environments associated with this system partition
	+ Other attributes on https://reframe-hpc.readthedocs.io/en/stable/regression\_test\_api.html#reframe.core.systems.SystemPartition
+ `current_environ` an object containing info of the programming environment that the regression test is currently executing with (usually provided in the config file)
   Usage:
	+ `current_environ.cc` string containing the C compiler
	+ `current_environ.cflags` list of string containing C compiler flags
	+ `current_environ.cxx` string containing the C++ compiler
	+ `current_environ.cxxflags` list of string containing C++ compiler flags
	+ `current_environ.cppflags` list of string containing preprocessor flags
	+ `current_environ.ldflags` list of string containing linker flags

### Test Validation part
For each test, one must define a function that determines whether the test passed or not and provide
a custom message that should be displayed when the test fails. For the sake of standardization, we do this within the 
following member function
  ```
    @sanity_function
    def validate_test(self):
    ...
    ...
    return reframe.utility.sanity.assert_true(hasTestPassed, msg) 
  ```
The developer must assign `hasTestPassed` to True if the test passed, or else assign it to False. Additionally, the developer
must assign a custom message to the variable `msg` that can be displayed when the test fails. 

__NOTE__: In order to help in parsing an output from test and comparing it with benchmark values, we have provided two files: Parser.py and CompareUtil.py (see below for details) that can be imported in the ReFrame test.py file. How to import?
  ```
  parser = rfm.utility.import_module_from_file("Parser.py")
  cu = rfm.utility.import_module_from_file("CompareUtil.py")
  ```
The above loads the Parser.py and CompareUtil.py from the directory containing the test.py and aliases them to ```parser``` and ```cu```, respectively. 

### Resource allocation part

# Test Utils
In order to help in parsing the output of a test and comparing it against some benchmark values, we have provided two util files:
+ Parser.py
+ CompareUtil.py

## Parser.py
It is a simple parser class to match a given string in a file and return the associated values. One can create an object in the following two ways:
+ ```parserObj = Parser.fromFilename(filename)```, where it takes a filename and parses the content of the file
+ ```parserObj = Parser.fromString(stringData)```, where it takes a string and parses the string

The most useful aspect of this class is the ```extactKeyValues``` member function
```
extractKeyValues(self, key, dtype='float', patternToCapture = r".*",
                         groupId = 0, dtypePattern = None):
```
The function arguments are
+ `key`: [Input] string to match
+ `dtype`: [Input] A string specifying the datatype of the values to be returned. 
_Default_: 'float'. Allowed values: 'int', 'float', 'complex', 'bool'.
+ `patternToCapture`: [Input] A string containing the regular-expression to capture after the `key` string. 
_Default_: captures everything after `key` until end of line
+ groupId: The group Id to capture from the pattern matched using `patterToCapture`. 
_Default_: 0 (i.e., uses the entire string matched after the `key`)
+ `dtypePattern`: A regular expression that defines how to extract a value of `dtype` from a given string (i.e., how to identify int, float, complex, bool from a string). 
_Default_: None (i.e., we use in-built regular expressions based on the `dtype`). This should work for most cases while dealing with int, float, complex, and bool datatypes.
+ `return` list of list where the outer list is the index of the occurence of `key` in the file and the inner list contains the list of `dtype` values extracted from a given occurence of `key`

## CompareUtil.py
__Requires__ ```numpy``` in the python environment.
The CompareUtil.py provides the ```Compare``` class that takes in two list of values and performs various comparison on them. It relies on numpy to perform some of the comparison. The most useful aspect of this class is the ```cmp``` member function
  ```
  cmp(self, val1, val2, tol = 1.0e-16, cmpType = 'absolute', normType="L2"):

  ```
The function arguments are:
+ `val1`: [Input] The first list
+ `val2`: [Tnput] The second list
+ `tol`: [Input] A float value specifying tolerance to which val1 and val2 (or their norms) should be matched to be deemed the same.
  _Default_: 1e-16
+ `cmpType`: [Input] String defining the type of comparison (absolute or  relative). The valid values are: 'absolute', 'relative'.
_Default_: 'absolute'
+ `normType`: [Input] String containing the vector-norm to apply to perform the comparison between the two lists. The Valid values are: "L1", "L2", "inf", "point". The normType="point" performs element wise comparison of the two lists and checks if the maximum absolute difference for any element is below the `tol`.
_Default_: "L2"
+ `return` Returns a pair `areComparable, msg` where `areComparable` is True when val1 and val2 are deemed the same (based on the `tol`, `cmpType`, and `normType` provided) and `msg` is a message that contains useful info when the two lists are not comparable


__TO DO <Ian>__
+ Add config instructions for ReFrame
+ Added Resources instructions for ReFrame 
+ CMake with multiple targets
+ Test Parser.py and CompareUtil.py
