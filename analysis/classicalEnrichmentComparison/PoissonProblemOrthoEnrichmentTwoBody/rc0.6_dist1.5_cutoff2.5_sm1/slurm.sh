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
# mpirun -n 56 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param10x10x10Uniformfe7Tol1e-8.in
# mpirun -n 56 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param11x11x11Uniformfe7Tol1e-8.in
# mpirun -n 56 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param12x12x12Uniformfe7Tol1e-8.in
# mpirun -n 56 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param13x13x13Uniformfe7Tol1e-8.in
# mpirun -n 56 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param14x14x14Uniformfe7Tol1e-8.in
# mpirun -n 56 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param15x15x15Uniformfe7Tol1e-8.in
# mpirun -n 56 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param16x16x16Uniformfe7Tol1e-8.in
# mpirun -n 56 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param17x17x17Uniformfe7Tol1e-8.in
# mpirun -n 56 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param18x18x18Uniformfe7Tol1e-8.in
# mpirun -n 56 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param19x19x19Uniformfe7Tol1e-8.in
# mpirun -n 56 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param19x19x19Uniformfe7Tol1e-8.in
# mpirun -n 56 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param20x20x20Uniformfe7Tol1e-8.in
mpirun -n 64 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param21x21x21Uniformfe7Tol1e-8.in
mpirun -n 64 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param22x22x22Uniformfe7Tol1e-8.in
mpirun -n 64 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param23x23x23Uniformfe7Tol1e-8.in
mpirun -n 64 --oversubscribe ./TestPoissonProblemEnrichment	PoissonProblemOrthoEnrichmentTwoBody/paramfiles/param24x24x24Uniformfe7Tol1e-8.in

