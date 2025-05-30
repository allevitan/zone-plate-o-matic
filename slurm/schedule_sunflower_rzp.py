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
parser.add_argument('grating_level', type=float, help='The fraction of the max amplitude to saturate at. Setting this to 1 provides best quality, at the expense of efficiency. 0.6 is usually a reasonable middle ground, and 0 produces a zone plate without any amplitude variation.')
parser.add_argument('buttress_width', type=float, help='The width of the buttresses, in nm. 0 will produce no buttresses')
parser.add_argument('--gpu', action='store_true', help='Whether to use the GPUs for the design phase.')
parser.add_argument('--n_processes', '-np', type=int, default=6, help='The number of processes to allocated to the realization stage, default of 6.')
parser.add_argument('--design-time','-dt', type=str, default=None, help='Time limit per design step. Default is default for the day queue on cpu or default for the gpu-week queue on gpu.')
parser.add_argument('--realize-time','-rt', type=str, default=None, help='Time limit per design step. Default is default for the day queue.')
parser.add_argument('--skip-design', action='store_true', help='If set, skips the design step and goes straight to realize')
parser.add_argument('--rect',  action='store_true', help='If True, returns a gds file with rectangles instead of polygons')

args = parser.parse_args()

jobarray = '--array=0-%d' % (args.n_zps - 1)

print(jobarray)

if args.gpu:
    design_command = 'design_sunflower_rzp_gpu.sh'
else:
    design_command = 'design_sunflower_rzp_cpu.sh'

print('Scheduling the design steps:')

design_job = [
    'sbatch',
    jobarray,
    design_command,
    args.plan_file
]

if not args.skip_design:
    slurm_output = subprocess.check_output(design_job).decode()

    job_id = int(slurm_output.split(' ')[-1])

    print('\nDesign steps have Job ID', job_id,'\n')
    print('Scheduling the realization steps')
else:
    print('Skipping design step, starting directly with realization')
    
design_folder = args.plan_file[:-8:] + '_intermediate_design'

if not args.skip_design:
    realize_dependency = '--dependency=afterok:'+str(job_id)
else:
    realize_dependency = '--dependency='

    
realize_job = [
    'sbatch',
    jobarray,
    '--cpus-per-task=' + str(args.n_processes),
    realize_dependency,
    'realize_sunflower_design.sh',
    design_folder,
    str(args.grating_level),
    str(args.buttress_width),
    str(args.n_processes),
    '--rect',
]

if args.rect:
    realize_job = realize_job + ['--rect']
    

slurm_output = subprocess.check_output(realize_job).decode()
job_id = int(slurm_output.split(' ')[-1])

print('Realization steps have Job ID', job_id, '\n')

realization_folder = args.plan_file[:-8:] \
                     + ('_GL=%0.2f_BW=%0.2fnm'
                        % (args.grating_level, args.buttress_width))

collate_job = [
    'sbatch',
    '--dependency=afterok:'+str(job_id),
    'collate_sunflower_design.sh',
    realization_folder,
]
    
    
slurm_output = subprocess.check_output(collate_job).decode()
job_id = int(slurm_output.split(' ')[-1])

print('Collation step has Job ID', job_id, '\n')
