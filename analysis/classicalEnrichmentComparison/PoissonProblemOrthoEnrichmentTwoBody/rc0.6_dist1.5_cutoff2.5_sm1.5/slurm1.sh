#!/bin/bash
#SBATCH --job-name=avirup
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=36
#SBATCH --mem-per-cpu=5000mb
#SBATCH --account=vikramg1
#SBATCH --partition=standard
#SBATCH --mail-type=all
#SBATCH --mail-user=avirup@umich.edu

module load gcc python/3.10.4 mkl boost openmpi/4.1.6 cmake
#mpirun -n <total procs> ./app <arg1> <arg2> <arg3> ...
mpirun -n 56 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param20x20x20Uniformfe5Tol1e-10.in
mpirun -n 56 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param20x20x20Uniformfe3Tol1e-10.in