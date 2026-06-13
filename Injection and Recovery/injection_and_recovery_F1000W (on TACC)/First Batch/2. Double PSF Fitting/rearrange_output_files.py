from pathlib import Path
import shutil

base_folder = Path("/work/10875/andreavu/ls6/injection_and_recovery_F1000W_bad")
output_folder = base_folder/"Outputs"
psf_outputs_folder = output_folder/'doublepsf_fitting_outputs'

for file in output_folder.glob("row_num*"):  # glob looks at the current directory
    if file.is_file():
        trial_num = file.stem.split('_')[2]
        file_path = str(file.parent) + '/' + str(file.name)

        for folder in psf_outputs_folder.glob("row_num*"):
            folder_num = folder.name.split('_')[2]
            if folder.is_dir() and folder_num == trial_num:
                des_path = str(folder.parent) + '/' + str(folder.name) + '/' + str(file.name)
                print(file_path, des_path)
                try:
                    shutil.move(file_path, des_path)
                except FileNotFoundError:
                    print(f"File not found, skipping: {file_path}")
                    continue
