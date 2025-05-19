import os

import numpy as np



aa_dict = {
    0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I',
    10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y',
    19: 'V', 20: '-'
}

def query_to_fasta(msa, filename):
    with open(filename, 'w') as file:
            header = f">seq_main"
            seq = msa[0]
            print(seq)
            file.write(header + "\n")

            sequence = ''.join([aa_dict[val] for val in seq if val != 20])
            print(sequence)
            file.write(sequence + "\n")

def msa_to_fasta(msa_file, output_file):
    list = os.listdir(msa_file)
    list = [f.replace('.npz', '') for f in list]
    for name in list:
        npz_path = os.path.join(msa_file, f"{name}.npz")
        msa = np.load(npz_path)['msa']
        filename = os.path.join(output_file, f"{name}.fasta")
        query_to_fasta(msa, filename)

    return output_file

if __name__ == '__main__':
    import os

    # 指定存储 npz 文件的目录
    directory = "/storage/xiangcx/cb/graduation_project/test_CASP_CAMEO/ CASP13/casp13_npz/"

    # 获取目录下所有 npz 文件的文件名（去掉后缀）
    filenames = [f[:-4] for f in os.listdir(directory) if f.endswith('.npz')]

    # 指定输出 txt 文件路径
    output_file = "/home/xiangcx/cb/graduation_project/casp13_npz.txt"

    # 将文件名逐行写入 txt 文件
    with open(output_file, "w") as f:
        for name in filenames:
            f.write(name + "\n")

    print(f"文件名已保存至 {output_file}")
