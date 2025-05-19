import os
import shutil
import subprocess
import re

import pandas as pd


def get_tmscore_and_rmsd(pdb1_path, pdb2_path):
    """
    输入两个PDB文件路径，使用TMscore程序进行结构比较，返回TM-score和RMSD。
    """
    try:
        # 构造命令（加上 -seq 参数）
        cmd = ["/home/xiangcx/bin/TMscore", pdb1_path, pdb2_path, "-seq"]

        # 执行命令并捕获输出
        result = subprocess.run(cmd, capture_output=True, text=True)

        # 检查是否运行成功
        if result.returncode != 0:
            print("运行 TMscore 时出错：", result.stderr)
            return None, None

        output = result.stdout

        # 提取 RMSD
        rmsd_match = re.search(r"RMSD of\s+the common residues=\s+([\d.]+)", output)
        rmsd = float(rmsd_match.group(1)) if rmsd_match else None

        # 提取 TM-score
        tmscore_match = re.search(r"TM-score\s+=\s+([\d.]+)", output)
        tmscore = float(tmscore_match.group(1)) if tmscore_match else None

        print(f"TM-score: {tmscore}")
        print(f"RMSD: {rmsd}")
        return tmscore, rmsd

    except Exception as e:
        print("发生异常：", e)
        return None, None


# pdb1 = "/storage/xiangcx/cb/graduation_project/test_CASP_CAMEO/CASP13/casp13_pdb/T1001-D1.pdb"
# pdb2 = "/storage/xiangcx/cb/graduation_project/test_CASP_CAMEO/CASP13/my_model_pdb/T1001-D1/14_2_T1001-D1.pdb"
#
# get_tmscore_and_rmsd(pdb1, pdb2)


def get_best_model(target_pdb,model_dir,best_output_dir):
    os.makedirs(best_output_dir, exist_ok=True)

    best_tmscore = -1.0
    best_rmsd = None
    best_model_path = None

    for filename in os.listdir(model_dir):
        if filename.endswith(".pdb"):
            model_path = os.path.join(model_dir, filename)
            tmscore, rmsd = get_tmscore_and_rmsd(target_pdb, model_path)
            print(f"{filename} -> TM-score: {tmscore}, RMSD: {rmsd}")

            if tmscore is not None and tmscore > best_tmscore:
                best_tmscore = tmscore
                best_rmsd = rmsd
                best_model_path = model_path

    if best_model_path:
        print("\n  最佳模型：", os.path.basename(best_model_path))
        print(f"   TM-score = {best_tmscore}")
        print(f"   RMSD     = {best_rmsd}")

        shutil.copy(best_model_path, best_output_dir)
        return best_tmscore, best_rmsd
    else:
        print("没有找到任何有效的 TM-score 结果。")
        return None, None

def make_TM_RMSD(root_target_dir,root_dir,best_output_dir):
    names = [f[:-4] for f in os.listdir(root_target_dir) if f.endswith('.pdb')]
    output_csv_file = os.path.join(best_output_dir, "inference_results.csv")
    results = []
    for name in names:
        target_pdb = os.path.join(root_target_dir, f"{name}.pdb")
        model_dir = os.path.join(root_dir, name)
        tmscore, rmsd = get_best_model(target_pdb,model_dir,best_output_dir)
        results.append([name, tmscore, rmsd])
    df = pd.DataFrame(results, columns=["Name", "TMscore", "RMSD"])
    df.to_csv(output_csv_file, index=False)


if __name__ == "__main__":
    root_target_dir = "/storage/xiangcx/cb/graduation_project/test_CASP_CAMEO/CASP13/casp13_pdb/"
    make_TM_RMSD(root_target_dir,
                 "/storage/xiangcx/cb/graduation_project/paper_test/all_tricks/pdb/",
                 "/storage/xiangcx/cb/graduation_project/paper_test/all_tricks/best_pdb/")