import json
from pathlib import Path

# ==============================================================
# Helper to patch notebooks
# ==============================================================

def patch_notebook_01():
    nb_path = Path("notebooks/01_Train_Stage1_Classifier.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Patch Cell 4: Hardware-accurate AMP_DTYPE
    c4 = "".join(nb["cells"][4]["source"])
    old_amp = "AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16"
    new_amp = """if torch.cuda.is_available():
    major_cc = torch.cuda.get_device_capability()[0]
    # BFloat16 on Compute Capability >= 8 (L4, A100); Float16 on Turing (Tesla T4)
    AMP_DTYPE = torch.bfloat16 if major_cc >= 8 else torch.float16
else:
    AMP_DTYPE = torch.float32"""
    assert old_amp in c4, "old_amp not found in Notebook 01 Cell 4"
    c4 = c4.replace(old_amp, new_amp)
    nb["cells"][4]["source"] = [l + "\n" for l in c4.split("\n")]

    # 2. Patch Cell 6: Robust patch count check (< 100 instead of == 0) and clean download
    c6 = "".join(nb["cells"][6]["source"])
    old_check = "if count_patches(DATASET_DIR / \"train\") == 0:"
    new_check = "if count_patches(DATASET_DIR / \"train\") < 100:"
    assert old_check in c6, "old_check not found in Notebook 01 Cell 6"
    c6 = c6.replace(old_check, new_check)
    nb["cells"][6]["source"] = [l + "\n" for l in c6.split("\n")]

    # 3. Patch Cell 14: Safe numpy conversion with .float()
    c14 = "".join(nb["cells"][14]["source"])
    old_np = """        probs_np = probs.detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()"""
    new_np = """        probs_np = probs.detach().float().cpu().numpy()
        targets = labels.detach().float().cpu().numpy()"""
    assert old_np in c14, "old_np not found in Notebook 01 Cell 14"
    c14 = c14.replace(old_np, new_np)
    nb["cells"][14]["source"] = [l + "\n" for l in c14.split("\n")]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook 01 successfully updated!")


def patch_notebook_02():
    nb_path = Path("notebooks/02_Pseudo_Masks_Train_Stage2.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Patch Cell 4: AMP_DTYPE
    c4 = "".join(nb["cells"][4]["source"])
    old_amp = "AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16"
    new_amp = """if torch.cuda.is_available():
    major_cc = torch.cuda.get_device_capability()[0]
    AMP_DTYPE = torch.bfloat16 if major_cc >= 8 else torch.float16
else:
    AMP_DTYPE = torch.float32"""
    if old_amp in c4:
        c4 = c4.replace(old_amp, new_amp)
        nb["cells"][4]["source"] = [l + "\n" for l in c4.split("\n")]

    # Patch Cell 6: count_patches < 100
    c6 = "".join(nb["cells"][6]["source"])
    old_check = "if count_patches(DATASET_DIR / \"train\") == 0:"
    new_check = "if count_patches(DATASET_DIR / \"train\") < 100:"
    if old_check in c6:
        c6 = c6.replace(old_check, new_check)
        nb["cells"][6]["source"] = [l + "\n" for l in c6.split("\n")]

    # Patch Cell 8: .float().cpu().numpy()
    c8 = "".join(nb["cells"][8]["source"])
    old_cam_np = "cam = cam_norm.squeeze(0).cpu().numpy()"
    new_cam_np = "cam = cam_norm.squeeze(0).float().cpu().numpy()"
    if old_cam_np in c8:
        c8 = c8.replace(old_cam_np, new_cam_np)
        nb["cells"][8]["source"] = [l + "\n" for l in c8.split("\n")]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook 02 successfully updated!")


def patch_notebook_03():
    nb_path = Path("notebooks/03_Train_Proposed_MedSAM.ipynb")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Patch Cell 4: AMP_DTYPE
    c4 = "".join(nb["cells"][4]["source"])
    old_amp = "AMP_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16"
    new_amp = """if torch.cuda.is_available():
    major_cc = torch.cuda.get_device_capability()[0]
    AMP_DTYPE = torch.bfloat16 if major_cc >= 8 else torch.float16
else:
    AMP_DTYPE = torch.float32"""
    if old_amp in c4:
        c4 = c4.replace(old_amp, new_amp)
        nb["cells"][4]["source"] = [l + "\n" for l in c4.split("\n")]

    # Patch Cell 6: count_patches < 100
    c6 = "".join(nb["cells"][6]["source"])
    old_check = "if count_patches(DATASET_DIR / \"train\") == 0:"
    new_check = "if count_patches(DATASET_DIR / \"train\") < 100:"
    if old_check in c6:
        c6 = c6.replace(old_check, new_check)
        nb["cells"][6]["source"] = [l + "\n" for l in c6.split("\n")]

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("Notebook 03 successfully updated!")


if __name__ == "__main__":
    patch_notebook_01()
    patch_notebook_02()
    patch_notebook_03()
