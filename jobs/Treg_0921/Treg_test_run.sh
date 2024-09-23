#! /bin/bash
#$ -S /bin/bash       # Specifies the script language (bash)
#$ -cwd               # Run in the current working directory
#$ -N Treg_gpu      # Job name
#$ -q gpu.q           # Specify the GPU queue
#$ -l gpu=1           # Request 1 GPU
#$ -l h_rt=0:30:00   # Maximum run time (24 hours in this case)
#$ -l mem_free=32G    # Request memory (32GB here)
#$ -o output.log      # Output file
#$ -e error.log       # Error log file

# Load necessary modules (e.g., CUDA)
module load CBI miniforge3/24.3.0-0
module load cuda/11.8
conda activate cellpath
export CUDA_VISIBLE_DEVICES=$SGE_GPU
# Run your Python script
gpuprof=$(dcgmi group -c mygpus -a $SGE_GPU | awk '{print $10}')
dcgmi stats -g $gpuprof -e
dcgmi stats -g $gpuprof -s $JOB_ID
python Treg_test.py

dcgmi stats -g $gpuprof -x $JOB_ID
dcgmi stats -g $gpuprof -v -j $JOB_ID
dcgmi group -d $gpuprof
