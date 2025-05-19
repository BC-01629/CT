import os
import subprocess
import numpy as np

aa_dict = {
        0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G', 8: 'H', 9: 'I',
        10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S', 16: 'T', 17: 'W', 18: 'Y',
        19: 'V', 20: '-'
    }

def npz_to_fasta(msa, filename):
    with open(filename, 'w') as file:
        for i, seq in enumerate(msa):
            header = f">seq{i + 1}"
            file.write(header + "\n")
            sequence = ''.join([aa_dict[val] for val in seq if val != 20])
            print(sequence)
            file.write(sequence + "\n")

def get_cluster(input_dir, output_dir,similarity_threshold = 0.7):
    os.makedirs(output_dir, exist_ok=True)
    fasta_files = [f for f in os.listdir(input_dir) if f.endswith(".fasta")]

    for fasta_file in fasta_files:
        input_path = os.path.join(input_dir, fasta_file)
        output_path = os.path.join(output_dir, fasta_file.replace(".fasta", "_cdhit.fasta"))

        cmd = [
            "cd-hit",
            "-i", input_path,
            "-o", output_path,
            "-c", str(similarity_threshold),
            "-M", str(0),
            "-T", str(16)
        ]

        print(f"Processing: {fasta_file} → {output_path}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {fasta_file}: {e}")

    print("All CD-HIT jobs completed!")


def extract_cluster_centers(filename, output_npy):
    centers = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('>Cluster'):
                continue

            if line.strip().endswith('*'):
                seq_id = int(line.split('>seq')[1].split('...')[0])
                centers.append(seq_id - 1)

    centers_array = np.array(centers, dtype=np.int32)
    np.save(output_npy, centers_array)
    return centers_array

def get_cluster_centers(input_dir, output_dir):

    for file in os.listdir(input_dir):
        if file.endswith(".clstr"):
            name = file.split("_cdhit")[0]
            filename = os.path.join(input_dir, file)
            output_file = os.path.join(output_dir, f"{name}.npy")
            extract_cluster_centers(filename, output_file)

    return None

if __name__ == "__main__":
    input_dir = "/storage/xiangcx/cb/graduation_project/real_train_db/"
    output_path = "/storage/xiangcx/cb/graduation_project/center_npyfiles/"
    get_cluster_centers(input_dir, output_path)
