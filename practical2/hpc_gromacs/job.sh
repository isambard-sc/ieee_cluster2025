#!/bin/bash

#SBATCH --job-name=gromacs
#SBATCH --gpus=4
#SBATCH --time=00:05:00         
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --reservation=IEEE_Cluster_Tutorial
#SBATCH --exclusive

#export GMX_GPU_PME_DECOMPOSITION=1
#export GMX_FORCE_UPDATE_DEFAULT_GPU=true
export GMX_ENABLE_DIRECT_GPU_COMM=1

#export CONT=${SCRATCH}/gromacs-2025.1.sif
export CONT=${PROJECTDIR}/gromacs-2025.1.sif

export OMP_NUM_THREADS=4

singularity run --nv -B ${PWD}:/host_pwd --pwd /host_pwd "${CONT}" gmx grompp -f pme.mdp

singularity run --nv -B ${PWD}:/host_pwd --pwd /host_pwd "${CONT}" gmx mdrun -ntmpi 4 -ntomp ${OMP_NUM_THREADS} -noconfout -nsteps 10000 -nstlist 300 -nb gpu -update gpu -pme gpu -npme 1 -dlb no -v -gpu_id 0123
