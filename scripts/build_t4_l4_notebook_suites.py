import json
import re
from pathlib import Path

BASE_DIR = Path("notebooks")

def build_suites():
    specs = [
        ("01_Train_Stage1_Classifier.ipynb", "01"),
        ("02_Pseudo_Masks_Train_Stage2.ipynb", "02"),
        ("03_Train_Proposed_MedSAM.ipynb", "03"),
        ("04_Exhaustive_Benchmark_Survival.ipynb", "04"),
    ]

    for base_name, prefix in specs:
        base_path = BASE_DIR / base_name
        with open(base_path, "r", encoding="utf-8") as f:
            base_nb = json.load(f)

        # -------------------------------------------------------------
        # 1. BUILD T4 DEDICATED NOTEBOOK
        # -------------------------------------------------------------
        t4_nb = json.loads(json.dumps(base_nb))
        t4_name = base_name.replace(".ipynb", "_T4.ipynb")
        t4_path = BASE_DIR / t4_name

        # Update metadata for T4
        t4_nb["metadata"]["accelerator"] = "GPU"
        t4_nb["metadata"]["colab"] = {"gpuType": "T4", "provenance": []}

        # Update badge and title in Cell 0
        c0_text = "".join(t4_nb["cells"][0]["source"])
        c0_text = c0_text.replace(
            "01_Train_Stage1_Classifier.ipynb", "01_Train_Stage1_Classifier_T4.ipynb"
        ).replace(
            "02_Pseudo_Masks_Train_Stage2.ipynb", "02_Pseudo_Masks_Train_Stage2_T4.ipynb"
        ).replace(
            "03_Train_Proposed_MedSAM.ipynb", "03_Train_Proposed_MedSAM_T4.ipynb"
        ).replace(
            "04_Exhaustive_Benchmark_Survival.ipynb", "04_Exhaustive_Benchmark_Survival_T4.ipynb"
        )
        c0_text = c0_text.replace("Recommended Hardware:** NVIDIA L4 GPU", "Dedicated Hardware:** NVIDIA Tesla T4 GPU (16GB VRAM, Turing)")
        c0_text = c0_text.replace("Precision:** Mixed Precision BF16 (L4) or FP16 (T4)", "Precision:** Mixed Precision FP16 (Native Turing Tensor Cores)")
        t4_nb["cells"][0]["source"] = [l + "\n" for l in c0_text.split("\n")[:-1]]

        # Notebook 01 Specific T4 Tuning
        if prefix == "01":
            # Cell 4: Force FP16 on T4
            c4_text = "".join(t4_nb["cells"][4]["source"])
            t4_amp_code = """AMP_DTYPE = torch.float16  # Native Turing Tensor Core FP16 acceleration
print(f"[READY] Compute Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"[READY] Mixed Precision AMP Dtype: {AMP_DTYPE} (Optimized for Tesla T4 16GB VRAM)")"""
            c4_text = re.sub(r"if torch\.cuda\.is_available\(\):.*?print\(f\"\[READY\] Mixed Precision AMP Dtype: \{AMP_DTYPE\}\"\)", t4_amp_code, c4_text, flags=re.DOTALL)
            t4_nb["cells"][4]["source"] = [l + "\n" for l in c4_text.split("\n")[:-1]]

            # Cell 8: T4 Batch Size 16, num_workers 2
            c8_text = "".join(t4_nb["cells"][8]["source"])
            c8_text = c8_text.replace('BATCH_SIZE = 32 if torch.cuda.is_available() and "L4" in torch.cuda.get_device_name(0) else 16', "BATCH_SIZE = 16  # Optimal for Tesla T4 16GB VRAM")
            c8_text = c8_text.replace('NUM_WORKERS = 4 if sys.platform != "win32" else 0', 'NUM_WORKERS = 2  # Optimal for Colab Standard 2-vCPU VM')
            t4_nb["cells"][8]["source"] = [l + "\n" for l in c8_text.split("\n")[:-1]]

        # Notebook 02 Specific T4 Tuning
        elif prefix == "02":
            # Cell 4: Force FP16
            c4_text = "".join(t4_nb["cells"][4]["source"])
            t4_amp_code = """AMP_DTYPE = torch.float16  # Native Turing Tensor Core FP16
print(f"[READY] Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) [Tesla T4 Tier]")"""
            c4_text = re.sub(r"if torch\.cuda\.is_available\(\):.*?print\(f\"\[READY\] Device: \{DEVICE\}.*?\"\)", t4_amp_code, c4_text, flags=re.DOTALL)
            t4_nb["cells"][4]["source"] = [l + "\n" for l in c4_text.split("\n")[:-1]]

            # Cell 10 (Stage 2 training loader): batch size 12
            c10_text = "".join(t4_nb["cells"][10]["source"])
            c10_text = c10_text.replace("batch_size=16", "batch_size=12  # Memory-safe for DeepLabV3+ ResNet-101 on T4")
            t4_nb["cells"][10]["source"] = [l + "\n" for l in c10_text.split("\n")[:-1]]

        # Notebook 03 Specific T4 Tuning
        elif prefix == "03":
            # Cell 4: Force FP16
            c4_text = "".join(t4_nb["cells"][4]["source"])
            t4_amp_code = """AMP_DTYPE = torch.float16  # Native Turing Tensor Core FP16
print(f"[READY] Compute Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) [Tesla T4 Tier]")"""
            c4_text = re.sub(r"if torch\.cuda\.is_available\(\):.*?print\(f\"\[READY\] Compute Device: \{DEVICE\}.*?\"\)", t4_amp_code, c4_text, flags=re.DOTALL)
            t4_nb["cells"][4]["source"] = [l + "\n" for l in c4_text.split("\n")[:-1]]

            # Cell 12: batch size 8 for MedSAM on T4
            c12_text = "".join(t4_nb["cells"][12]["source"])
            c12_text = c12_text.replace("batch_size=16", "batch_size=8  # Memory-safe for MedSAM ViT on T4 16GB")
            t4_nb["cells"][12]["source"] = [l + "\n" for l in c12_text.split("\n")[:-1]]

        with open(t4_path, "w", encoding="utf-8") as f:
            json.dump(t4_nb, f, indent=1)
        print(f"Created dedicated T4 notebook: {t4_path}")

        # -------------------------------------------------------------
        # 2. BUILD L4 DEDICATED NOTEBOOK
        # -------------------------------------------------------------
        l4_nb = json.loads(json.dumps(base_nb))
        l4_name = base_name.replace(".ipynb", "_L4.ipynb")
        l4_path = BASE_DIR / l4_name

        # Update metadata for L4
        l4_nb["metadata"]["accelerator"] = "GPU"
        l4_nb["metadata"]["colab"] = {"gpuType": "L4", "provenance": []}

        # Update badge and title in Cell 0
        c0_text = "".join(l4_nb["cells"][0]["source"])
        c0_text = c0_text.replace(
            "01_Train_Stage1_Classifier.ipynb", "01_Train_Stage1_Classifier_L4.ipynb"
        ).replace(
            "02_Pseudo_Masks_Train_Stage2.ipynb", "02_Pseudo_Masks_Train_Stage2_L4.ipynb"
        ).replace(
            "03_Train_Proposed_MedSAM.ipynb", "03_Train_Proposed_MedSAM_L4.ipynb"
        ).replace(
            "04_Exhaustive_Benchmark_Survival.ipynb", "04_Exhaustive_Benchmark_Survival_L4.ipynb"
        )
        c0_text = c0_text.replace("Recommended Hardware:** NVIDIA L4 GPU", "Dedicated Hardware:** NVIDIA L4 GPU (24GB VRAM, Ada Lovelace, Colab Pro)")
        c0_text = c0_text.replace("Precision:** Mixed Precision BF16 (L4) or FP16 (T4)", "Precision:** Mixed Precision BF16 (Native Ada Lovelace Tensor Cores)")
        l4_nb["cells"][0]["source"] = [l + "\n" for l in c0_text.split("\n")[:-1]]

        # Notebook 01 Specific L4 Tuning
        if prefix == "01":
            # Cell 4: Force BF16 on L4
            c4_text = "".join(l4_nb["cells"][4]["source"])
            l4_amp_code = """AMP_DTYPE = torch.bfloat16  # Native Ada Lovelace Tensor Core BF16 acceleration
print(f"[READY] Compute Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
print(f"[READY] Mixed Precision AMP Dtype: {AMP_DTYPE} (Optimized for NVIDIA L4 24GB VRAM)")"""
            c4_text = re.sub(r"if torch\.cuda\.is_available\(\):.*?print\(f\"\[READY\] Mixed Precision AMP Dtype: \{AMP_DTYPE\}\"\)", l4_amp_code, c4_text, flags=re.DOTALL)
            l4_nb["cells"][4]["source"] = [l + "\n" for l in c4_text.split("\n")[:-1]]

            # Cell 8: L4 Batch Size 32, num_workers 4
            c8_text = "".join(l4_nb["cells"][8]["source"])
            c8_text = c8_text.replace('BATCH_SIZE = 32 if torch.cuda.is_available() and "L4" in torch.cuda.get_device_name(0) else 16', "BATCH_SIZE = 32  # Full throughput on NVIDIA L4 24GB VRAM")
            c8_text = c8_text.replace('NUM_WORKERS = 4 if sys.platform != "win32" else 0', 'NUM_WORKERS = 4  # High-throughput 4 workers on Colab Pro VM')
            l4_nb["cells"][8]["source"] = [l + "\n" for l in c8_text.split("\n")[:-1]]

        # Notebook 02 Specific L4 Tuning
        elif prefix == "02":
            # Cell 4: Force BF16
            c4_text = "".join(l4_nb["cells"][4]["source"])
            l4_amp_code = """AMP_DTYPE = torch.bfloat16  # Native Ada Lovelace Tensor Core BF16
print(f"[READY] Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) [NVIDIA L4 Tier]")"""
            c4_text = re.sub(r"if torch\.cuda\.is_available\(\):.*?print\(f\"\[READY\] Device: \{DEVICE\}.*?\"\)", l4_amp_code, c4_text, flags=re.DOTALL)
            l4_nb["cells"][4]["source"] = [l + "\n" for l in c4_text.split("\n")[:-1]]

            # Cell 10 (Stage 2 training loader): batch size 24
            c10_text = "".join(l4_nb["cells"][10]["source"])
            c10_text = c10_text.replace("batch_size=16", "batch_size=24  # High throughput for DeepLabV3+ ResNet-101 on L4 24GB")
            l4_nb["cells"][10]["source"] = [l + "\n" for l in c10_text.split("\n")[:-1]]

        # Notebook 03 Specific L4 Tuning
        elif prefix == "03":
            # Cell 4: Force BF16
            c4_text = "".join(l4_nb["cells"][4]["source"])
            l4_amp_code = """AMP_DTYPE = torch.bfloat16  # Native Ada Lovelace Tensor Core BF16
print(f"[READY] Compute Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}) [NVIDIA L4 Tier]")"""
            c4_text = re.sub(r"if torch\.cuda\.is_available\(\):.*?print\(f\"\[READY\] Compute Device: \{DEVICE\}.*?\"\)", l4_amp_code, c4_text, flags=re.DOTALL)
            l4_nb["cells"][4]["source"] = [l + "\n" for l in c4_text.split("\n")[:-1]]

            # Cell 12: batch size 16 for MedSAM on L4
            c12_text = "".join(l4_nb["cells"][12]["source"])
            c12_text = c12_text.replace("batch_size=16", "batch_size=16  # Full batch for MedSAM ViT on L4 24GB")
            l4_nb["cells"][12]["source"] = [l + "\n" for l in c12_text.split("\n")[:-1]]

        with open(l4_path, "w", encoding="utf-8") as f:
            json.dump(l4_nb, f, indent=1)
        print(f"Created dedicated L4 notebook: {l4_path}")

if __name__ == "__main__":
    build_suites()
