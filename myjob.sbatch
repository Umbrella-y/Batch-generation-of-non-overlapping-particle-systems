#!/bin/bash

#SBATCH --account=zhuyiying
#SBATCH --partition=hpib
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

export FI_PROVIDER=verbs

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load lammps/201812/lmp_intelcpu_intelmpi


srun -n $SLURM_NTASKS --cpu-bind=cores lmp_intelcpu_intelmpi -in in.8sinter_1.lmp -sf intel -pk intel 0 omp $OMP_NUM_THREADS mode double
