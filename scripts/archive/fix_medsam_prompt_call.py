import json
from pathlib import Path

nb03_path = Path("notebooks/03_Train_Proposed_MedSAM.ipynb")
with open(nb03_path, "r", encoding="utf-8") as f:
    nb03 = json.load(f)

c14_src = "".join(nb03["cells"][14]["source"])
old_block = """        # Extract online CAM point prompts from Stage 1 (forward_cam returns (cam, y))
        with torch.no_grad():
            if hasattr(model_stage1, "forward_cam"):
                cam_s1, probs_s1 = model_stage1.forward_cam(imgs)
                cam_up = torch.nn.functional.interpolate(cam_s1, size=(224, 224), mode="bilinear", align_corners=False)
                prompts = prompt_gen(cam_up, probs_s1)
            else:
                prompts = {"point_coords": None, "point_labels": None}
        
        optimizer_medsam.zero_grad()
        with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=(DEVICE.type == "cuda")):
            pred_masks = model_medsam(imgs, prompts["point_coords"], prompts["point_labels"])"""

new_block = """        # Extract online CAM point prompts from Stage 1 (forward_cam returns (cam, y))
        with torch.no_grad():
            if hasattr(model_stage1, "forward_cam"):
                cam_s1, probs_s1 = model_stage1.forward_cam(imgs)
                cam_up = torch.nn.functional.interpolate(cam_s1, size=(224, 224), mode="bilinear", align_corners=False)
                prompt_dict = prompt_gen(cam_up, probs_s1)
                primary_c = int(torch.argmax(probs_s1.mean(dim=0)).item()) if probs_s1 is not None else 0
                point_coords = prompt_dict[primary_c]["point_coords"]
                point_labels = prompt_dict[primary_c]["point_labels"]
            else:
                point_coords, point_labels = None, None
        
        optimizer_medsam.zero_grad()
        with torch.amp.autocast('cuda', dtype=AMP_DTYPE, enabled=(DEVICE.type == "cuda")):
            pred_masks = model_medsam(imgs, point_coords, point_labels)"""

assert old_block in c14_src, "Target block not found in Notebook 03 Cell 14"
c14_src = c14_src.replace(old_block, new_block)
nb03["cells"][14]["source"] = [l + "\n" for l in c14_src.split("\n")]
with open(nb03_path, "w", encoding="utf-8") as f:
    json.dump(nb03, f, indent=1)
print(f"Successfully fixed MedSAM prompts in {nb03_path} Cell 14!")
