# Practical 1- Finding your way around Isambard-AI

This session allows users to familiarise themselves with Isambard-AI, including:

* Logging in and understanding the role of `clifton`
* Basic use of the `slurm` workload manager
* Build and run some simple examples

All activities in this practical session are based on examples in the [documentation](https://docs.isambard.ac.uk).

## Logging in via SSH

Prior to the workshop, you should have completed the [required pre-workshop steps](https://docs.isambard.ac.uk/training/ieee_cluster2025/#required-pre-workshop-steps).

SSH access to Isambard-AI is controlled using certificates using a command line tool called `clifton` and you will need to set this up before SSHing onto the platform. Note that [binary versions](https://github.com/isambard-sc/clifton/releases/tag/0.2.0) of are available. **Take care to select the correct architecture.**

Follow the [Login Guide](https://docs.isambard.ac.uk/user-documentation/guides/login/) to enable to you to SSH into Isambard-AI into the project created specifically for this tutorial:

`ssh t5c.aip2.isambard`

## Modules

Modules provide a method for providing a range of basic software and libraries.

Use `module avail` to see what is provides, and then `module list` to see what is presently loaded. `module load` adds modules to those already loaded.

**Task 1**

We will use the `osu_hello` command from the OSU MPI microbenchmark suite as an example of how modules make executables available.

* Try running `osu_hello`. What happens? 
* Inspect the available modules and find the one mentioning 'osu'. Load it and run `osu_hello` again.

**Task 2**

This time, we will use modules to gain access to a newer version of a compiler than the one installed by default. On Isambard-AI, this is especially important to access full support for the ARM Grace CPU.

* Run `gcc --version` and make a note of the version.
* Then run `module load PrgEnv-gnu` and rerun the previous command. What's changed?

Find out more about modules in the [Modules and Compilers guide](https://docs.isambard.ac.uk/user-documentation/guides/modules/)

## Slurm workload manager

Isambard-AI uses Slurm to manage the workload on the compute nodes and will be familiar to many HPC users. This section sets some short tasks to demonstrate basic usage and encourage research in the documentation.

The [Slurm guide](https://docs.isambard.ac.uk/user-documentation/guides/slurm/) provides the information you need to complete the tasks below.

**To minimise queueing, please use `--reservation=IEEE_Cluster_Tutorial` during this session.**

**Task 1**

Set up and run a job submission script to return the hostname and GPU information for a single GPU job. Then, modify this script to run on 8 GPUs concurrently.

**Task 2**

Use `srun` to run the `hostname` command on a single GPU. Start an interactive session on a single GPU and again run `hostname`. How are these two approaches different?


## Option 1- Python

Look at the [Python guide](https://docs.isambard.ac.uk/user-documentation/guides/python/).

**Suggested task** 

Install conda and create an environment in which you install `scipy`. Write a Slurm job submission script to run a simple script, for example:

```
from scipy import constants

print(constants.liter)
```

## Option 2- Spack

Look at the [spack guide](https://docs.isambard.ac.uk/user-documentation/guides/spack/).

**Suggested task** 

Install spack and the Isambard `buildit` repository. Create an environment and install `osu-micro-benchmarks`. Use `srun` to run `osu_bw` on two nodes to measure the inter-node bandwidth.

Extension activity: Instead of using Spack, run `osu_bw` using the preinstalled module. How can you control which version you are running?

## Option 3- Containers

Look at the [containers guide](https://docs.isambard.ac.uk/user-documentation/guides/containers/).

**Suggested task**

Use the documentation to run the `lolcow` container on both the login node and a compute node.

Extension activity: Run the `nvidia-smi --list-gpus` command via a container. What is different when you run on either a compute node or the login node?

