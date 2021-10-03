# Instruction to using ReFrame

## ReFrame Guide
https://reframe-hpc.readthedocs.io/en/stable/

## Table of contents <a name="contents"></a>
+ [Installation](#installation)
+ [How to run](#howtorun)
+ [Config](#config)
  + [`systems`](#systems):
  + [`environments`](#environments):
  + [`logging`](#logging):
+ [ReFrame Regression Test Basics](#reframeregression)
  + [Constructor part](#constructor)
  + [Compiler setting part](#compiler)
  + [Resource allocation part](#resourceallocation)
  + [Test Validation part](#testvalidation)

[Test Utils](#testutils)

## Installation <a name="installation"></a> 
+ Download ReFrame to a directory (preferably put it in dftefe/test directory)
  ```shell
  cd dftefe/test
  git clone https://github.com/eth-cscs/reframe.git
  ```
+ Make sure that you have a python3.7 or higher available in your PATH 
(e.g., `module load python3.7` or `module load python 3.9`)

+ Change directory to ReFrame directory and run the ./bootstrap.sh script
  ```shell
  cd reframe
  ./bootstrap.sh
  ```
+ Verify the installation by running the following to display the version
  ```shell
  ./bin/reframe -V
  ```
[back to top](#contents)

## How to run <a name="howtorun"></a>
Under the test folder, execute `reframe -C ./config/mysettings.py -c ./test.py -r`

[back to top](#contents)

## Config <a name="config"></a>
While there are multiple ways to set the configuration file, we stick to the command in [How to run](#howtorun): 
define a big variable in a python file called `mysettings.py` and stored in `${PROJECT_HOME}/test/config`. In this 
instruction, useful features for testing DFT-EFE is introduced. A more elaborative tutorial can be found 
[here](https://reframe-hpc.readthedocs.io/en/stable/configure.html?highlight=site_configuration#) and a verbose 
explanation to variable config files can be found [here](https://reframe-hpc.readthedocs.io/en/stable/config_reference.html). 

The big variable storing the configuration information is split into three parts:
+ [`systems`](#systems):
+ [`environments`](#environments): 
+ [`logging`](#logging):

A minimal example of mysettings.py looks like

[back to top](#contents)

### systems <a name="systems"></a>
`systems` is used to define different machines (local computer, mac, greatlakes, summit, cori), some useful attributes are
+ `name` (__Required__): the name of the machine (__Required__), e.g, `'name': 'localhost'`, `'name': 'greatlakes'`, etc.
+ `dscr` (optional): description of the machine (optional)
+ `hostnames` (__Required__): 
+ `launcher` (__Required__): 
+ `modules_system` (optional): The module system on the machine used to manage `module load`, a full list of supported 
  module system in 
  ReFrame can be found [here](https://reframe-hpc.readthedocs.io/en/stable/config_reference.html#.systems[].modules_system).
+ `partition` (__Required__): 
    ```python
    {
        'name': 'compute',
        'scheduler': 'slurm',
        'launcher': 'srun',
        'access': ['-A vikramg1'],
        'environs': ['builtin', 'gnu']
    }
    ```
  + `name`(__Required__): the name of the partition/queue 
  + `scheduler`(__Required__): job scheduler used by the machine, currently support `local`, `oar`, `pbs`, `sge`, `slurm`, 
    `squeue`, `torque`
  + `launcher`(__Required__): ReFrame supports various types of parallel launchers, e.g, `srun`, `local`, `mpirun`, 
    `ibrun`. A full list of ReFrame supported launcher can be found at [here](https://reframe-hpc.readthedocs.io/en/stable/config_reference.html#systems-.partitions-.launcher). 
  + `access` (optional): A list of job scheduler options to be passed to the job script generator. This is where the 
    charging account should be specified. e.g, `'access': ['-A vikramg1']`.
  + `environs: (optional)` The [environments](#environments) defined in [`environments`](#environments) used to be 
    run on this partition.

[back to top](#contents)

### environments <a name="environments"></a>
`environments` is used to determine the environment configurations in ReFRame. The running and compilation environment 
for the test is set up through this part. This piece is related to the `eviron` attribute in [`systems`](#systems). 
Some useful attributes and an example is provided for reference.
```python
{
    'name': 'intel',
    'modules': ['intel', 'mkl'],
    'variables': [['DFT_EFE_LINKER','"-mkl=parallel"']]
    'cc': 'icc',
    'cxx': 'icpc',
    'ftn': 'ifort',
    'target_systems': ['greatlakes'],
},
```
+ `name` (__Required__): the user defined name of the environment.
+ `modules` (optional): modules to be loaded from `module load`. The modules will be loaded using the module 
  management system specified by `modules_system` attribute in [`systems`](#systems).
+ `variables` (optional): the environmental variables can be set here for `cmake` and compilation.
+ `cc` `cxx` `ftn` (optional): Specify the compilers used to compile the test for C, C++, or Fortran respectively.   
  _Default_: `cc`, `CC`, `ftn`
+ `target_systems` (optional): Specify which systems set in `systems` are allowed to run the test using this 
  environment.
  _Default_: `["*"]` (all available systems.)
+ `cppflags`, `cflags`, `cxxflags`, `fflags` (optional): A list of flags to be used with the specified environment by 
  default for C preprocessor, C, C++, or Fortran. Be careful that `cppflags` is for C preprocessor instead of C++.
+ `ldflags` (optional): A list of linker's flags to be used with the specified environment by default.

[back to top](#contents)

### logging <a name="logging"></a>
We suggest using the default setting in `./config/mysettings.py`.

[back to top](#contents)

## ReFrame Regression Test Basics <a name="reframeregression"></a>
Although ReFrame provides many features, we will be using a limited number of features. Each test must be a python class. Typically, a test class should be
split into four parts: 
+ A [constructor part](#constructor) which sets the important attributes (source dir, build system type, etc)
+ A [compiler flag setter](#compiler) to set any compilation flag or environment   
+ A [test validation part](#testvalidation) which decides whether the test passed or failed and define a message to display if the 
  test fails
+ A [resource allocation part](#resourceallocation) which sets the resources needed by the test and the launcher options via a queueing system

[back to top](#contents)

### Constructor part <a name="constructor"></a>
The following are the usual attributes that one requires to set in each test class. The ones marked as __Required__ 
are the mandatory ones that must be specified for each test class.
+ `descr` (Optional): A string containing description of the test 
+ `valid_systems` (__Required__): A list of 'system:partition' on which to run the test  
   Examples:  
    + `valid_systems = ['*']` to specify all the system:paritions defined in the config file
    + `valid_systems = ['greatlakes:*']` to specify all the partitions for the greatlakes system
    + `valid_systems = ['greatlakes:compute', 'cori:login']` to specify greatlakes compute parition 
       and cori login parition 
   __Recommended__: `valid_systems = ['*']` in which case it selects all the systems:partitions defined in the config file 
+ `valid_prog_environs` (__Required__): A list of programming environment on which to run the test 
  Examples:
    + `valid_prog_environs = ['*']` to specify all the programming environments defined in the config file
    + `valid_prog_environs = ['gnu', 'intel']`  to use both gnu and intel environments, as defined in the config file  
  __Recommended__: `valid_prog_environs = ['*']`
    + `sourcepath`: A string containing path to source file. Use it when you have a single source file to test 
      (__Required__ if `sourcesdir` is not set)  
       Examples: `source_path = 'test.cc'`
+ `sourcesdir`: A string containing the path to the source directory. Use it when you have multiple source files 
   (__Required__ if `sourcepath` is not set) 
+ `executable_opts` (Optional) : A string containing the command line arguments to the test executable
  Examples:
    + `executable_opts = '> outfile'` to redirect the `stdout` to outfile
        + `executable_opts = 'arg1 arg2'` to provide arg1 and arg2 as command line input 

+ `tagsDict` (__Required__): A dictionary specifying various attributes to the test.  As a standard convention, we use 5 keys, each of which are allowed certain possible values. These key:value pairs are then use to elp us runonly tests matching certain tag(s). The four keys and their their possible values are:
    1. Key: `'compileOrRun'`. Possible values: `'compile'`, `'run'`. It determines whether the test is a 
	a. compile only test: A test which only tests the compilation of the test sources, in which case `'compileOrRun': 'compile'`; or
        b. a run test: A test which compiles and runs the test, in which case (i.e., `'compileOrRun': 'run'`)
    2. Key: `'unitOrAggregate'`. Possible values: `'unit'`, `'aggregate'`. It determines whether a test is a 
	a. unit test: It tests only a single function, in which case `'unitOrAggregate': 'unit'`; or
        b. aggregate test: It tests a set of functions or a class, in which case `'unitOrAggregate': 'aggregate'`
    3. Key: `'slowOrFast'`. Possible values: `'slow'`, `'fast'`. It determines whether a test is slow (i.e., it takes more than 30 seconds, in which case `'slowOrFast': 'slow'`) or fast (i.e., it takes less than 30 seconds, in which case `'slowOrFast': 'fast'`).
    4. Key: `'arch'`: Possible values: `'cpu'`,`'gpu'`, `'both'`. It determines whether the test is to be run on a cpu (in which case ', gpu, or both architectures.  	
A user should populate the tagsDict with 
    # the appropriate values for each of the four keys: 'compileOrRun',
    # 'unitOrAggregate', 'arch', 'serialOrParallel'
    tagsDict = {'compileOrRun': 'compile', 'unitOrAggregate':
                'aggregate', 'slowOrFast': 'fast', 'arch': 'cpu',
                'serialOrParallel': 'serial'}
+ `build_system` (Optional): A string to define the build system.
   Example: 
    + `build_system = 'CMake'` to indicate CMake based build 
   (__Recommended__): build_system = 'CMake'. We will be using a CMake build system for most of our tests
    
+ `make_opts` (Optional): A list of string which provides additional options to be used along with `make` command while compiling a test. 
    The most common use case is when the CMakeLists defines multiple targets to be built, but the test requires only one or more of them to be built. 
    By default it will build all the targets specified in the CMakeLists. In that case you can define the selected list of targets to be built through make_opts, i.e.,
    make_opts = ['target-name']

+ `extra_resources` a dictionary setting resources that should be allocated to the test by the scheduler. 
   Typically, the resources are defined in the configuration of a system partition, e.g,  
  ```python
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
  ```python
  self.extra_resources = {
    'gpu': {'num_gpus_per_node': 2}
    'cpu': {
        'memory': '4GB',
        'time': '48:00:00'
    }
  }
  ```
    This setup will generate a submission script based on the backend scheduling system. For `slurm`, an example output 
of the generated header for the above example is
    ```shell
    #SBATCH --gres=gpu:2
    #SBATCH --mem=4GB
    #SBATCH --time=48:00:00
    ```
It is possible to replace `#SBATCH` for specific lines. Refer to [extra_resources](https://reframe-hpc.readthedocs.io/en/stable/regression_test_api.html?highlight=extra_resource#reframe.core.pipeline.RegressionTest.extra_resources)
page on ReFrame website. Please also refer to [here](https://reframe-hpc.readthedocs.io/en/stable/regression_test_api.html?highlight=extra_resource#mapping-of-test-attributes-to-job-scheduler-backends) the scheduler specific attribute mapping.

[back to top](#contents)

### Compiler setting part  <a name="compiler"></a>
For each test, one might need to set certain compilation and environment flags. We do this within the following member function
  ```python
  @run_before('compile')
  def set_compiler_flags(self):
  ```
The ```@run_before('compile')``` ReFrame decorator marks the ```set_compiler_flags(self)``` function as 
something that 
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

[back to top](#contents)

### Test Validation part <a name="testvalidation"></a>
For each test, one must define a function that determines whether the test passed or not and provide
a custom message that should be displayed when the test fails. For the sake of standardization, we do this within the 
following member function
  ```python
    @sanity_function
    def validate_test(self):
    ...
    ...
    return reframe.utility.sanity.assert_true(hasTestPassed, msg) 
  ```
The developer must assign `hasTestPassed` to True if the test passed, or else assign it to False. Additionally, the developer
must assign a custom message to the variable `msg` that can be displayed when the test fails. 

__NOTE__: In order to help in parsing an output from test and comparing it with benchmark values, we have provided two files: Parser.py and CompareUtil.py (see below for details) that can be imported in the ReFrame test.py file. How to import?
  ```python
  parser = rfm.utility.import_module_from_file("Parser.py")
  cu = rfm.utility.import_module_from_file("CompareUtil.py")
  ```
The above loads the Parser.py and CompareUtil.py from the directory containing the test.py and aliases them to ```parser``` and ```cu```, respectively. 

[back to top](#contents)

### Resource allocation part <a name="resourceallocation"></a>

[back to top](#contents)

# Test Utils <a name="testutils"></a>
In order to help in parsing the output of a test and comparing it against some benchmark values, we have provided two util files:
+ [Parser.py](#parser)
+ [CompareUtil.py](#compareutil)

[back to top](#contents)

## Parser.py <a name="parser"></a>
It is a simple parser class to match a given string in a file and return the associated values. One can create an object in the following two ways:
+ ```parserObj = Parser.fromFilename(filename)```, where it takes a filename and parses the content of the file
+ ```parserObj = Parser.fromString(stringData)```, where it takes a string and parses the string

The most useful aspect of this class is the ```extactKeyValues``` member function
```python
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

[back to top](#contents)

## CompareUtil.py <a name="compareutil"></a>
__Requires__ ```numpy``` in the python environment.
The CompareUtil.py provides the ```Compare``` class that takes in two list of values and performs various comparison on them. It relies on numpy to perform some of the comparison. The most useful aspect of this class is the ```cmp``` member function
```python
cmp(self, val1, val2, tol = 1.0e-16, cmpType = 'absolute', normType="L2")
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

[back to top](#contents)

__TO DO <Ian>__
+ ~~Add config instructions for ReFrame~~
+ Added Resources instructions for ReFrame 
+ ~~CMake with multiple targets~~
+ Test Parser.py and CompareUtil.py
