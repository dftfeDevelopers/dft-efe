# Copyright 2016-2021 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

#
# Generic fallback configuration
#

# cmake_flags = ["-DMPI_C_COMPILER=mpicc",
#                "-DMPI_CXX_COMPILER=mpic++",
#                '''-DDFTEFE_BLAS_LIBRARIES="-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"''',
#                '''-DDFTEFE_SCALAPACK_LIBRARIES="-L${MKLROOT}/lib/intel64 -lmkl_scalapack_lp64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -lgomp -lpthread -lm -ldl"''',
#                "-DBUILD_DOCS=OFF",
#                '''-DCMAKE_PREFIX_PATH="/home/vikramg/DFT-FE-softwares/dealiiDevCustomized/install_gcc8.2.0_openmpi4.0.6_minimal"''',
#                "-DENABLE_CUDA=OFF"]

site_configuration = {
    'systems': [
        {
            'name': 'greatlakes',
            'descr': 'Greatlakes UMICH',
            'hostnames': ['.*'],
            'modules_system': 'lmod',
            'partitions': [
                {
                    'name': 'standard',
                    'scheduler': 'slurm',
                    'launcher': 'srun',
                    'access': ['-A vikramg1'],
                    'environs': ['gnu'],
                    'resources': [
                        {
                            'name': 'cpu',
                            'options': ['--partition=standard',
                                        '--time={time_limit}',
                                        '--nodes={num_nodes}',
                                        '--ntasks-per-node={num_tasks_per_node}',
                                        '--ntasks={ntasks}',
                                        '--mem-per-cpu={mem_per_cpu}']
                        }
                    ]
                },
                {
                    'name': 'gpu',
                    'scheduler': 'slurm',
                    'launcher': 'srun',
                    'access': ['-A vikramg1'],
                    'environs': ['gnu'],
                    'resources': [
                        {
                            'name': 'gpu',
                            'options': ['--partition=gpu',
                                        '--time={time_limit}',
                                        '--nodes={num_nodes}',
                                        '--gpus-per-node={gpus_per_node}'
                                        '--ntasks-per-node={num_tasks_per_node}',
                                        '--ntasks={ntasks}',
                                        '--mem-per-cpu={mem_per_cpu}']
                        }
                    ]
                },
                {
                    'name': 'interactive',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['builtin'],
                    'resources': [
                        {
                            'name': 'cpu',
                            'options': ['']
                        }
                    ]
                }
            ]
        },
        {
            'name': 'generic',
            'descr': 'Generic example system',
            'hostnames': ['.*'],
            'partitions': [
                {
                    'name': 'default',
                    'scheduler': 'local',
                    'launcher': 'local',
                    'environs': ['builtin']
                }
            ]
        },
    ],
    'environments': [
        {
            'name': 'gnu',
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran',
            'modules': [],
            'target_systems': ['greatlakes']
            #'variables': [['DFT_EFE_LINKER','"-L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl"']]
        },
        {
            'name': 'builtin',
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran',
            'target_systems': ['greatlakes']
        },
        {
            'name': 'builtin',
            'cc': 'cc',
            'cxx': '',
            'ftn': ''
        },
    ],
    'logging': [
        {
            'handlers': [
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s'
                },
                {
                    'type': 'file',
                    'level': 'debug',
                    'format': '[%(asctime)s] %(levelname)s: %(check_info)s: %(message)s',  # noqa: E501
                    'append': False
                }
            ],
            'handlers_perflog': [
                {
                    'type': 'filelog',
                    'prefix': '%(check_system)s/%(check_partition)s',
                    'level': 'info',
                    'format': (
                        '%(check_job_completion_time)s|reframe %(version)s|'
                        '%(check_info)s|jobid=%(check_jobid)s|'
                        '%(check_perf_var)s=%(check_perf_value)s|'
                        'ref=%(check_perf_ref)s '
                        '(l=%(check_perf_lower_thres)s, '
                        'u=%(check_perf_upper_thres)s)|'
                        '%(check_perf_unit)s'
                    ),
                    'append': True
                }
            ]
        }
    ],
}
