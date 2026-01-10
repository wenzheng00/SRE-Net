import os
import subprocess
import argparse

def run_cmd(cmd, env=None):
    print(f"\n▶ 运行命令：{cmd}\n")
    result = subprocess.run(cmd, shell=True, executable="/bin/bash", env=env)
    if result.returncode != 0:
        raise RuntimeError("❌ 命令执行失败")

def extract_hippocampus_mask(nii_path, subject_id, freesurfer_home, subjects_dir, output_dir):
    # 配置环境变量
    env = os.environ.copy()
    env["FREESURFER_HOME"] = freesurfer_home
    env["SUBJECTS_DIR"] = subjects_dir

    setup_cmd = f"source {freesurfer_home}/SetUpFreeSurfer.sh"

    # 创建目标文件夹
    mgz_path = os.path.join(subjects_dir, subject_id, "mri", "orig", "001.mgz")
    os.makedirs(os.path.dirname(mgz_path), exist_ok=True)

    # 步骤1：转换 NIfTI 为 mgz
    run_cmd(f"{setup_cmd} && mri_convert {nii_path} {mgz_path}", env=env)

    # 步骤2：执行 FreeSurfer 重建（包含分割）
    run_cmd(f"{setup_cmd} && recon-all -s {subject_id} -autorecon1 -autorecon2", env=env)

    # 步骤3：提取左右海马体掩码（17 = 左海马，53 = 右海马）
    aseg_path = os.path.join(subjects_dir, subject_id, "mri",  "orig", "aseg.mgz")
    os.makedirs(output_dir, exist_ok=True)
    lh_out = os.path.join(output_dir, f"{subject_id}_lh_hippo.nii.gz")
    rh_out = os.path.join(output_dir, f"{subject_id}_rh_hippo.nii.gz")

    run_cmd(f"{setup_cmd} && mri_binarize --i {aseg_path} --match 17 --o {lh_out}", env=env)
    run_cmd(f"{setup_cmd} && mri_binarize --i {aseg_path} --match 53 --o {rh_out}", env=env)

    print("✅ 海马提取完成：")
    print(f"  左海马: {lh_out}")
    print(f"  右海马: {rh_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FreeSurfer 脑MRI海马分割")
    parser.add_argument("--nii", required=True, help="输入MRI路径（.nii或.nii.gz）")
    parser.add_argument("--subject", required=True, help="FreeSurfer中的subject ID")
    parser.add_argument("--fs_home", required=True, help="FreeSurfer安装路径")
    parser.add_argument("--fs_subjects", required=True, help="FreeSurfer SUBJECTS_DIR路径")
    parser.add_argument("--output", required=True, help="输出mask保存路径")

    args = parser.parse_args()

    extract_hippocampus_mask(
        nii_path=args.nii,
        subject_id=args.subject,
        freesurfer_home=args.fs_home,
        subjects_dir=args.fs_subjects,
        output_dir=args.output
    )
