# Instruction to using ReFrame

## ReFrame Guide
https://reframe-hpc.readthedocs.io/en/stable/

## Run Test in greatlakes
cd <Test Folder containing testfilename>
 ../reframe/bin/reframe -C ../config/mysettings.py -c <testfilename> -r -p=gnu -n='BuildAnd*'
## Run Test in perlmutter
cd <Test Folder containing testfilename>
../reframe/bin/reframe -C ../config/mysettings.py -c <testfilename> -r -p=PrgEnv-gnu -n='BuildAnd*'

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
While there are multiple ways to set the configuration file, stick to the command in [How to run](#howtorun) and 
define a big list named `site_configuration` in a python file called `mysettings.py` in `${PROJECT_HOME}/test/config`. In this 
instruction, useful features for testing DFT-EFE is introduced. A more elaborative tutorial can be found 
[here](https://reframe-hpc.readthedocs.io/en/stable/configure.html?highlight=site_configuration#) and a verbose 
explanation to variable config files can be found [here](https://reframe-hpc.readthedocs.io/en/stable/config_reference.html). 

For our purpose, the `site_configuration` list should contain the following three parts:
+ [`systems`](#systems):
+ [`environments`](#environments): 
+ [`logging`](#logging):

A minimal example of mysettings.py looks like

[back to top](#contents)

### systems <a name="systems"></a>
`'systems'` is used to define different machines (local computer, mac, greatlakes, summit, cori), some useful attributes are
+ `'name'` (__Required__): A string containing the name of the machine, e.g., `'name': 'localhost'`, `'name': 'greatlakes'`, etc. 
  This name is meant for the user to identify the machine through a meaningful name, which can then be used inside the body of a test to select a certain machine. 
  Internally, ReFrame identifies the machine through the `hostnames` value (see below). To elaborate, one might refer to greatlakes cluster at the University of 
  Michigan as `'name':'greatlakes'`. However, the actual hostname might be of the form gl-login1.arc-ts.umich.edu for a login node or gl-1234.arc-ts.umich.edu 
  for a compute node with ID 1234.  
+ `'dscr'` (optional): description of the machine (optional)
+ `'hostnames'` (__Required__): A list of strings (including any regular expressions) that can be used to identify the machine. 
  Internally, ReFrame fetches the hostname by running the `hostname` Linux command and tallying the output against the list of strings provided
  in `'hostnames'`.
  __Recommended__: __Do not__ use a catch-all regular expression like `'hostnames'=['.*']`. Otherwise ReFrame will match the first such system 
  listed in the config file, which need not be the machine you are working on. Instead, run the `hostname` command from your machine and create a 
  generic regular expression that would only select that hostname. For example, on greatlakes given that the hostnames are of the form 
  'gl-login1.arc-ts.umich.edu' or 'gl-1234.arc-ts.umich.edu', one can use `'hostnames'=['gl.*arc-ts.umich.edu']` to select them.
+ `'launcher'` (Optional): A string containing the launcher. For example, `srun`, `mpirun`, or just the local launcher (`./`). 
  The __standard practice__ we follow is to provide the launcher for a specific partition (see below in `partition`).   
+ `'modules_system'` (Optional): A string specifying the the module system on the machine used to manage `module load`.
  For example, for Lua based module, use `'module_system'='lmod'`A full list of supported module system in ReFrame can be found [here](https://reframe-hpc.readthedocs.io/en/stable/config_reference.html#.systems[].modules_system).
+ `'partition'` (__Required__): A dictionary that specifies a partition in the system (machine). For example, a machine can have separate login, compute partitions, 
  or it can have multiple partitions based on the architecture (cpu, gpu, hybrid partitions, etc.). A system partition in ReFrame is not bound to a real scheduler partition. 
  It is a virtual partition or separation of the system. The binding to a real scheduler partition happens through the resource allocation options within a partition 
  (see the `'resources'` attribute below).
  A minimal example for a partition is given below 
  ```python
  'partition' = {
    'name': 'compute',
    'scheduler': 'slurm',
    'launcher': 'srun',
    'access': ['-A vikramg1'],
    'environs': ['builtin', 'gnu']
    'resources': 
      [
        {
          'name': 'cpu',
          'options': ['--partition=standard',
          '--time={time_limit}',
          '--nodes={num_nodes}',
          '--ntasks-per-node={num_tasks_per_node}',
          '--mem-per-cpu={mem_per_cpu}']
        }
      ]
    
  }
  ```
  The important keys in the partition dictionary are: 
  + `name`(__Required__): A string containing name of the partition. This name is meant for the user to identify the partition through a meaningful name. 
    The __standard practice__ we follow is to include the string 'gpu' for any GPU based partition.
  + `scheduler`(__Required__): job scheduler used by the machine, currently ReFrame supports `local`, `oar`, `pbs`, `sge`, `slurm`, 
    `squeue`, `torque`.
  + `launcher`(__Required__): ReFrame supports various types of parallel launchers, e.g, `srun`, `local`, `mpirun`, 
    `ibrun`. A full list of ReFrame supported launcher can be found at [here](https://reframe-hpc.readthedocs.io/en/stable/config_reference.html#systems-.partitions-.launcher). 
  + `'access'` (Optional): A list of job scheduler options to be passed to the job script generator. This is where the charging account should be specified.  
     For example, `'access': ['-A vikramg1']`. If you are running the test on a cluster, it is very likely that you will need to set the access to able to launch any job.
  + `'environs': (Optional)` A list of strings defining the programming environments on which the test must be run. The actual configuration of these environments are defined in the 
    `environments` section (see [environments](#environments))
  + `'resources':` (Optional) A list of dictionaries which specifies the resource allocation options (specific to a queueing system like slurm, pbs). Additionally, 
    it can provide any placeholders that can be specified in the test file. For example, for Slurm, we can define two separate dictionaries named 'cpu' and 'gpu' 
    each with different resource allocation options,
    ```python
      'resources': 
       [{
	  'name': 'cpu',
          'options': ['--partition=standard',
          '--time={time_limit}',
          '--nodes={num_nodes}',
          '--ntasks-per-node={num_tasks_per_node}',
          '--mem-per-cpu={mem_per_cpu}']
        },
        {
          'name': 'gpu',
          'options': ['--partition=gpu',
          '--time={time_limit}',
          '--nodes={num_nodes}',
          '--gpus-per-node={gpus_per_node}'
          '--ntasks-per-node={num_tasks_per_node}',
          '--mem-per-cpu={mem_per_cpu}']
        }
      ]
    ```
    In that case, the a test file can specify the resource options using the `extra_resources` attribute as follows: 
    ```python
    self.extra_resources{
      'cpu': {'time_limit': time_limit,
            'num_nodes': num_nodes,
            'num_tasks_per_node': num_tasks_per_node,
            'mem_per_cpu': mem_per_cpu
           },
      'gpu': {
            'time_limit': time_limit,
            'num_nodes': num_nodes,
            'num_tasks_per_node': num_tasks_per_node,
            'mem_per_cpu': mem_per_cpu,
            'gpus_per_node': gpus_per_node
    	   }
     }
    ```
    In the above, the `'name'` defined in the `'resources'` is for the user to identify the resources. The queueing system only relies on the `--partition` options to decide the cpu or gpu resources.  
    

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
+ `variables` (Optional): the environmental variables can be set here for `cmake` and compilation. 
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
Although ReFrame provides many features, we will be using a limited number of features. Each test must be:
1. a python class,
2. be decorated with `@reframe.simple_test`, and
3. inherit from `reframe.RegressionTest` or `reframe.CompileOnlyRegressionTest`, for run or compile only tests, respectively,  

Typically, a test class should be split into four parts: 
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

+ `tagsDict` (__Required__): A dictionary specifying various attributes to the test.  As a standard convention, we use 5 keys, each of which are allowed certain possible values. These key:value pairs help us filter and run tests matching certain tag(s). The five keys and their their possible values are:
    1. Key: `'compileOrRun'`. Possible values: `'compile'`, `'run'`. It determines whether the test is a:
        1. compile only test: A test which only tests the compilation of the test sources, in which case `'compileOrRun': 'compile'`; or
        2. a run test: A test which compiles and runs the test, in which case (i.e., `'compileOrRun': 'run'`) 
    2. Key: `'unitOrAggregate'`. Possible values: `'unit'`, `'aggregate'`. It determines whether a test is a:
    	1. unit test: It tests only a single function, in which case `'unitOrAggregate': 'unit'`; or
        2. aggregate test: It tests a set of functions or a class, in which case `'unitOrAggregate': 'aggregate'`
    3. Key: `'slowOrFast'`. Possible values: `'slow'`, `'fast'`. It determines whether a test is: 
    	1. slow: A test that takes more than 30 seconds, in which case `'slowOrFast': 'slow'`; or 
    	2. fast: A test that takes less than 30 seconds, in which case `'slowOrFast': 'fast'`)
    4. Key: `'arch'`: Possible values: `'cpu'`,`'gpu'`, `'both'`. It determines whether the test is to be run on a:
    	1. cpu: A test that should run only on cpu(s), in which case `'arch':'cpu'`; or
    	2. gpu: A test that should run only on gpu(s), in which case `'arch':'gpu'`; or
        3. both: A test that should be run on both cpu(s) and gpu(s), in which case `'arch':'both'`
    5. Key: `serialOrParallel`. Possible values: `'serial'`, `'parallel'`. It determines whether the test is to be run on:
	1. serial: A serial test which does not require any dependence on mpi or openmp, in which case `'serialOrParallel':'serial`; or
	2. parallel: A parallel test which depends on mpi or openmp, in which case `serialOrParallel: 'parallel'` 	

    A user should populate the tagsDict with  the appropriate values for each of the five keys.
    Example: 
    ```
    tagsDict = {'compileOrRun': 'compile', 'unitOrAggregate':
                'aggregate', 'slowOrFast': 'fast', 'arch': 'cpu',
                'serialOrParallel': 'serial'}
    ```
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
something that should run prior to any compilation. The following are commonly used attributes that one might need to set in this function
+ `self.build_system.make_opts` (Optional): list of strings to be passed as command line options to cmake 
  Typically, this should be used to specify any target for cmake. For example, 
  if the CMakeLists contains various targets and we want to test only one of them, 
  we can provide the target here. e.g., `self.build_system.make_opts` = ['all'] 
  __Recommended__: We define the options in the constructor of the class using the `make_opts` variable (see `make_opts` in the [Constructor part of a test](#constructor)). Subsequently, 
  we assign `self.build_system.make_opts=self.make_opts`   
+ `build_system` attributes (Optional): There are several build system attributes (e.g., compiler type, compiler flags, linker flags, etc) that one can set within the ReFrame test class. However, in most circumstances, _one should avoid_ setting these attributes within a ReFrame test class. As far as possible, these attribues should be set globally for  all tests within the config file. 
  Important `build_system` attributes are:
	+ `self.build_system.cc` string to specify the C compiler
	+ `self.build_system.cflags` list of string to specify the C compiler flags
	+ `self.build_system.cxx` string to specify the C++ compiler
	+ `self.build_system.cxxflags` list of string to specify the C++ compiler flags
	+ `self.build_system.cppflags` list of string to specify the preprocessor flags
	+ `self.build_system.ldflags` list of string to specify the linker flags
	+ `self.build_system.nvcc` string to specify the CUDA compiler

+ `current_system`: An object containing info of the the system the regression test is currently executing on
   Usage:
	+ `self.current_system.descr` a string containing the description of the system (usually provided in the config file)
	+ `self.current_system.name` a string containing the name of the system
	+ `self.current_system.paritions` a list of string containing the partitions in the system
	+ More attributes on https://reframe-hpc.readthedocs.io/en/stable/regression\_test\_api.html#reframe.core.systems.System
+ `current_partition`: An object containing info of the system partition the regression test is currently executing on (usually provided in the config file)
  Usage:
	+ `curren_partition.name` a string containing the name of the partition
	+ `current_partition.descr` a string containing the description of the parition
	+ `current_partition.access` list of string containing scheduler options for accessing this system partition 
	+ `current_partition.environs` list of string containing the programming environments associated with this system partition
	+ Other attributes on https://reframe-hpc.readthedocs.io/en/stable/regression\_test\_api.html#reframe.core.systems.SystemPartition
+ `current_environ`: An object containing info of the programming environment that the regression test is currently executing with (usually provided in the config file)
   Usage:
	+ `self.current_environ.cc` string containing the C compiler
	+ `self.current_environ.cflags` list of string containing C compiler flags
	+ `self.current_environ.cxx` string containing the C++ compiler
	+ `self.current_environ.cxxflags` list of string containing C++ compiler flags
	+ `self.current_environ.cppflags` list of string containing preprocessor flags
	+ `self.current_environ.ldflags` list of string containing linker flags

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
The developer __must__ assign `hasTestPassed` to True if the test passed, or else assign it to False. Additionally, the developer __must__ assign a custom message to the variable `msg` that can be displayed when the test fails. 

__NOTE__: In order to help in parsing an output from test and comparing it with benchmark values, we have provided two files: Parser.py and CompareUtil.py that can be imported in the ReFrame test.py file. How to import?
  ```python
  parser = rfm.utility.import_module_from_file("Parser.py")
  cu = rfm.utility.import_module_from_file("CompareUtil.py")
  ```
The above loads the Parser.py and CompareUtil.py from the directory containing the test.py and aliases them to ```parser``` and ```cu```, respectively. See below the [Test Utils](#testutils) section for more details about the Parser.py and CompareUtil.py.

[back to top](#contents)

### Resource allocation part <a name="resourceallocation"></a>

[back to top](#contents)

# Test Utils <a name="testutils"></a>
In order to help in parsing the output of a test and comparing it against some benchmark values, we have provided two util files:
+ [Parser.py](#parser): To parse the output of a test or any file/string
+ [CompareUtil.py](#compareutil): To compare two list of values based on user-defined tolerance and vector-norm
+ [setupSystems.py](#setupsystems): To filter valid system:partition pairs and allocate resources based on user inputs.

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
_Default_: "inf"
+ `return` Returns a tuple of `areComparable, norm, msg` where `areComparable` is True when val1 and val2 are deemed the same (based on the `tol`, `cmpType`, and `normType` provided), `norm` is the difference between val1 and val2 based on the `normType`, and `msg` is a message that contains useful info when the two lists are not comparable

## SetupSystems.py <a name="setupsystems"></a>
The setupSystems.py provides two functions to help delegate the task of selecting the valid systems to run the test on and the allocation of resources for the test. 
+ ```getValidSystems``` function:
  ```python
    def getValidSystems(key):
  ```
  where the function parameters and return values are:
  + `key`: A string containing either of the following values: `'cpu'`, `'gpu'`, or `'both'`. It defines the architecture type on which the test is supposed to be run.
  + `return`: A list of string of the format 'system:partition' based on the `key`  
   
  The function fetches all the system:partition pairs from the config file (or the command line argument) and then filters them based on the `key`. The convention is for `key='cpu'`, we select only those system:partitions that do not contain the string 'gpu' in it. Conversely, for `key = 'gpu'`, only those system:partitions are included which contains the string 'gpu'. For `key='both'`, all the system:partiions are selected.

+ The ```setResources``` function:

  ```python
  def setResources(archTag = 'both', time_limit = "00:02:00", num_nodes = 1, num_tasks_per_node = 1, mem_per_cpu =
                 '2gb', gpus_per_node = 1):
   ``` 
  where the function parameters and return values are:
  + `archTag` string that can be 'cpu', 'gpu', or 'both'. It defines the architecture type on which the test is supposed to be run (__Default__: 'both')
  + `time_limit`: string of the format "hrs:mins:secs". It defines the maximum wall time for the test. (__Default__: "00:02:00")
  + `num_nodes`: integer for number of nodes to allocate (__Default__: 1)
  + `num_tasks_per_node`: integer for number of tasks to use per node (__Default__: 1)
  +  `mem_per_cpu`: string of the format "<number>mb" or "<number>gb" to define the memory to allocate per cpu (__Default__: "2gb")
  +  `gpus_per_node`: integer for number of gpus to allocate per node. This is used only when the `archTag='cpu'` or `archTag='both'`. (Default: 1) 
  + `return`: A dictionary containing the key:value pairs required by the `'resources'` list in the config file. To elaborate, `'resources'` list is defined in the config file for a system partition, e.g,  
    ```python
      'resources': 
       [{
	  'name': 'cpu',
          'options': ['--partition=standard',
          '--time={time_limit}',
          '--nodes={num_nodes}',
          '--ntasks-per-node={num_tasks_per_node}',
          '--mem-per-cpu={mem_per_cpu}']
        },
        {
          'name': 'gpu',
          'options': ['--partition=gpu',
          '--time={time_limit}',
          '--nodes={num_nodes}',
          '--gpus-per-node={gpus_per_node}'
          '--ntasks-per-node={num_tasks_per_node}',
          '--mem-per-cpu={mem_per_cpu}']
        }
      ]
    ```
    
    In that case, the setupResources returns the following dictionary:
    ```python
    return {
      'cpu': {'time_limit': time_limit,
            'num_nodes': num_nodes,
            'num_tasks_per_node': num_tasks_per_node,
            'mem_per_cpu': mem_per_cpu
           },
      'gpu': {
            'time_limit': time_limit,
            'num_nodes': num_nodes,
            'num_tasks_per_node': num_tasks_per_node,
            'mem_per_cpu': mem_per_cpu,
            'gpus_per_node': gpus_per_node
    	   }
     }
    ```
    where `time_limit`, `num_nodes`, `num_tasks_per_node`, `mem_per_cpu`, and `gpus_per_node` are defined through the input parameters to the function. 

  The setResources() function is typically called from the ```set_launcher_and_resources()``` function of a test (see the [resource allocation part of test](#resourceallocation)). For example,
  ```python
    @run_before('run')
      def set_launcher_and_resources(self):
        self.extra_resources = ss.setResources(archTag = 'gpu', time_limit = "02:00:00", num_nodes = 1, num_tasks_per_node = 2, mem_per_cpu = "3gb", gpus_per_node = 2)
  ```
  where ```ss = rfm.utility.import_module_from_file("setupSystems.py")``` (i.e., an alias for ```setupSystems```). This setup will generate a submission script based on the backend scheduling system. For `slurm`  it will generate 
    ```shell
    #SBATCH --time=02:00:00
    #SBATCH --nodes=1
    #SBATCH --num_tasks_per_node=2
    #SBATCH --mem=3GB
    #SBATCH --gres=gpu:2
  ```
  However, providing all these details in the test.py might make the body of the test too cluttered. Therefore, we delegate the task of resource allocation to setResources() which creates 

[back to top](#contents)

__TO DO <Ian>__
+ ~~Add config instructions for ReFrame~~
+ Added Resources instructions for ReFrame 
+ ~~CMake with multiple targets~~
+ Test Parser.py and CompareUtil.py
