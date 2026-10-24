import json
from pathlib import Path

nb_path = Path("01_Train_Stage1_Classifier_v2.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for idx, cell in enumerate(nb["cells"]):
    cell_type = cell.get("cell_type")
    source = "".join(cell.get("source", []))[:80].replace("\n", " ")
    outputs = cell.get("outputs", [])
    print(f"\n--- Cell {idx} [{cell_type}]: {source} ---")
    if outputs:
        for out_idx, out in enumerate(outputs):
            out_type = out.get("output_type")
            print(f"  Output {out_idx} [{out_type}]:")
            if "text" in out:
                text = "".join(out["text"])
                print(f"    Text: {text[:300]}...")
            if "traceback" in out:
                tb = "".join(out["traceback"])
                print(f"    Traceback:\n{tb}")
            if "ename" in out:
                print(f"    Error: {out['ename']}: {out.get('evalue')}")
