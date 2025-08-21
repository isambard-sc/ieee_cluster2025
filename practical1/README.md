# Practical 1- Finding your way around Isambard-AI

This session aims to allow users to find their feet on Isambard-AI and explore the options for building and running software.

### Aims

* Log in to Isambard-AI and explain the role of `clifton`
* Basic use of the `slurm` workload manager
* Build and run some simple examples
* Signpost to relevant documentation

## Logging in via SSH

Prior to the workshop, you should have completed the [required pre-workshop steps](https://docs.isambard.ac.uk/training/ieee_cluster2025/#required-pre-workshop-steps). This will have created your account on the system.

SSH access to Isambard-AI is controlled using certificates using a command line tool called `clifton`. Follow the [Login Guide](https://docs.isambard.ac.uk/user-documentation/guides/login/).

SSH into Isambard-AI into the project created specifically for this session:

`ssh t5c.aip2.isambard`

You should now be logged in to one of the login nodes.

## Modules

Modules provide a method for providing a limited range of basic software and libraries.

Use `module avail` to see what is provides, and then `module list` to see what is presently loaded. `module load` adds modules to those already loaded.

Task 1
* Try running `osu_hello`. What happens?
* Inspect the available modules and find the one mentioning 'osu'. Load it and run `osu_hello` again.

Task 2
* Run `gcc --version`
* Run `module load PrgEnv-gnu` and try again. What's changed?

Find out more about modules in the [Modules and Compilers guide](https://docs.isambard.ac.uk/user-documentation/guides/modules/)

## Slurm workload manager

Isambard-AI uses Slurm to manage the workload on the compute nodes and will be familiar to many HPC users. This section sets two short tasks to demonstrate basic usage and encourage research in the documentation.

The [Slurm guide](https://docs.isambard.ac.uk/user-documentation/guides/slurm/) provides the information you need to complete the tasks below.

Note that to minimise queueing, please use `--reservation=IEEE_Cluster_Tutorial` during this session.

Task 1
* Set up and run a job submission script to return the hostname and GPU information for a single GPU job.
* Modify this script to run on 8 GPUs concurrently

Task 2
* Use `srun` to run the `hostname` command on a single GPU.
* Start an interactive session on a single GPU and again run `hostname`.


## Option 1- Conda/Python

Work through the [Conda guide](https://docs.isambard.ac.uk/user-documentation/guides/python/)

## Option 2- Spack

Work through the [spack guide](https://docs.isambard.ac.uk/user-documentation/guides/spack/)

## Option 3- Containers

Work through the [containers guide](https://docs.isambard.ac.uk/user-documentation/guides/containers/)

