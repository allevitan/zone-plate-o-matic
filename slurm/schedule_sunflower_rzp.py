import subprocess
import argparse


parser = argparse.ArgumentParser(
    prog='schedule_sunflower_rzp',
    description='Submits a bunch of slurm jobs to make a complete sunflower-style randomized zone plate .gds output from a plan file, with the proper job dependencies.')

# TODO: read the number of the zps from the plan file. I don't do this
# now because importing h5py fucks up MKL in some way that causes all the
# slurm jobs to fail.
parser.add_argument('plan_file', type=str, help='The plan file to use.')
parser.add_argument('n_zps', type=int, help='The total number of zps in the plan file')
parser.add_argument('--gpu', action='store_true', help='Whether to use the GPUs')
parser.add_argument('--design-time','-dt', type=str, default=None, help='Time limit per design step. Default is default for the day queue on cpu or default for the gpu-week queue on gpu.')
parser.add_argument('--realize-time','-rt', type=str, default=None, help='Time limit per design step. Default is default for the day queue.')

args = parser.parse_args()

jobarray = '--array=0-%d' % (args.n_zps - 1)

print(jobarray)

if args.gpu:
    design_command = 'design_sunflower_rzp_gpu.sh'
else:
    design_command = 'design_sunflower_rzp_cpu.sh'

print('Scheduling the design steps:')

design_job = ['sbatch',
              jobarray,
              design_command,
              args.plan_file]

print(design_job)
#exit()
slurm_output = subprocess.check_output(design_job)

print(slurm_output)
exit()

job_id = int(slurm_output.split(' ')[-1])

print('Design steps have Job ID', job_id,'\n')
print('Scheduling the realization steps')
