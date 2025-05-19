import os
import subprocess
from time import sleep

from sbatch import gen_sbatch
# N是重复次数
def run_trRosetta(N=5,base_npz = '"/storage/xiangcx/cb/graduation_project/test_CASP_CAMEO/ CASP13/casp13_pred_npz/1_model/pred_T0950-D1.npz"',
                  base_fasta = '"/storage/xiangcx/cb/graduation_project/test_CASP_CAMEO/ CASP13/casp13_fasta/T0950-D1.fasta"',
                  base_out = '"/storage/xiangcx/cb/graduation_project/test_CASP_CAMEO/ CASP13/trRosetta_pdb/1_model/"',out_name='pred_T0950-D1',
                  options = '-m 2 -r no-idp --orient',start_id=1):
    base_command = '/home/xiangcx/anaconda3/envs/pytorch/bin/python3 "/home/xiangcx/trRosettaX_single/trRosetta.py"'
    try:
        os.makedirs(base_out[1:-1]+out_name,exist_ok=True)
    except FileExistsError:
        pass
    base_out = f'{base_out[:-1]}{out_name}/{out_name}{{}}.pdb"'
    for i in range(start_id, N + start_id):
        out_file = base_out.format(i)
        command = f'{base_command} -NPZ {base_npz} -FASTA {base_fasta} -OUT {out_file} {options}'
        print(command)
        # subprocess.run('conda activate pytorch',shell=True)
        # subprocess.run(command, shell=True)
        print(f'Executed: {command}')

if __name__=='__main__':
    total = 15
    run_num = 3
    py = '/home/xiangcx/anaconda3/envs/pytorch/bin/python3'
    base_npz_file = "/storage/xiangcx/cb/graduation_project/paper_test/all_tricks/"
    base_fasta_file = "/storage/xiangcx/cb/graduation_project/test_CASP_CAMEO/CASP13/casp13_fasta/"
    base_out = "/storage/xiangcx/cb/graduation_project/paper_test/all_tricks/pdb/"
    names = [f[:-6] for f in os.listdir("/storage/xiangcx/cb/graduation_project/test_CASP_CAMEO/CASP13/casp13_fasta/") if f.endswith('.fasta')]
    for j in range(total):
        for name in names:
            try:
                os.makedirs(os.path.join(base_out, name), exist_ok=True)
            except FileExistsError:
                pass
            for i in range(run_num):
                npz_path = os.path.join(base_npz_file,f"{name}.npz")
                fasta_path = os.path.join(base_fasta_file,name+'.fasta')
                out_path = os.path.join(base_out,name,f"{j+2}_{i}_{name}.pdb")
                options = '-m 2 -r no-idp --orient'
                task = f'/home/xiangcx/trRosettaX_single/trRosetta.py -NPZ {npz_path} -FASTA {fasta_path} -OUT {out_path} {options}'
                print(task)
                gen_sbatch(job_name=f'Rosetta_{j}_{i}_{name}', py=py,
                           task=task,
                           node=None,
                           output_path="/storage/xiangcx/cb/sbatch_out/",
                           err_path="/storage/xiangcx/cb/sbatch_err/",
                           output_dir="/storage/xiangcx/cb/sbatch_script/",
                           mode='cpu', run=True, delete=True)
                sleep(3)
        sleep(3000)








