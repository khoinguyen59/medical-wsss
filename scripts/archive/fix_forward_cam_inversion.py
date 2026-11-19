import json
from pathlib import Path

# ==========================================
# 2. FIX NOTEBOOK 02: Cell 8
# ==========================================
nb02_path = Path("notebooks/02_Pseudo_Masks_Train_Stage2.ipynb")
with open(nb02_path, "r", encoding="utf-8") as f:
    nb02 = json.load(f)

c8_src = "".join(nb02["cells"][8]["source"])
old_cam_extract = """        # Extract CAM
        if hasattr(model_stage1, "forward_cam"):
            logits, cam = model_stage1.forward_cam(img_tensor)
            cam = torch.softmax(cam, dim=1).squeeze(0).cpu().numpy()
        else:
            logits = model_stage1(img_tensor)
            # Fallback surrogate CAM
            cam = np.zeros((5, 224, 224), dtype=np.float32)
            cam[4, :, :] = 0.5
        
        # Construct 5-class distribution (TUM, STR, LYM, NEC, BACK)
        if cam.shape[0] == 4:
            bg_channel = np.clip(1.0 - np.sum(cam, axis=0, keepdims=True), 0.0, 1.0)
            full_cam = np.concatenate([cam, bg_channel], axis=0)
        else:
            full_cam = cam"""

new_cam_extract = """        # Extract CAM (forward_cam returns (cam, y) where cam is (B, 4, 28, 28))
        if hasattr(model_stage1, "forward_cam"):
            cam_raw, cls_probs = model_stage1.forward_cam(img_tensor)
            cam_up = torch.nn.functional.interpolate(cam_raw, size=(224, 224), mode="bilinear", align_corners=False)
            cam_min = cam_up.view(1, 4, -1).min(dim=-1)[0].view(1, 4, 1, 1)
            cam_max = cam_up.view(1, 4, -1).max(dim=-1)[0].view(1, 4, 1, 1)
            cam_norm = (cam_up - cam_min) / (cam_max - cam_min + 1e-5)
            cam = cam_norm.squeeze(0).cpu().numpy()  # (4, 224, 224)
        else:
            cam = np.zeros((4, 224, 224), dtype=np.float32)

        # Construct 5-class distribution (TUM, STR, LYM, NEC, BACK)
        bg_channel = np.clip(1.0 - np.sum(cam, axis=0, keepdims=True), 0.0, 1.0)
        full_cam = np.concatenate([cam, bg_channel], axis=0)  # (5, 224, 224)"""

assert old_cam_extract in c8_src, "old_cam_extract not found in Notebook 02 Cell 8"
c8_src = c8_src.replace(old_cam_extract, new_cam_extract)
nb02["cells"][8]["source"] = [l + "\n" for l in c8_src.split("\n")]
with open(nb02_path, "w", encoding="utf-8") as f:
    json.dump(nb02, f, indent=1)
print(f"Successfully updated {nb02_path} Cell 8!")

# ==========================================
# 3. FIX NOTEBOOK 03: Cell 14
# ==========================================
nb03_path = Path("notebooks/03_Train_Proposed_MedSAM.ipynb")
with open(nb03_path, "r", encoding="utf-8") as f:
    nb03 = json.load(f)

c14_src = "".join(nb03["cells"][14]["source"])
old_medsam_prompt = """        # Extract online CAM point prompts from Stage 1
        with torch.no_grad():
            if hasattr(model_stage1, "forward_cam"):
                logits_s1, cam_s1 = model_stage1.forward_cam(imgs)
                cam_norm = torch.softmax(cam_s1, dim=1)
                prompts = prompt_gen(cam_norm, torch.sigmoid(logits_s1))
            else:
                prompts = {"point_coords": None, "point_labels": None}"""

new_medsam_prompt = """        # Extract online CAM point prompts from Stage 1 (forward_cam returns (cam, y))
        with torch.no_grad():
            if hasattr(model_stage1, "forward_cam"):
                cam_s1, probs_s1 = model_stage1.forward_cam(imgs)
                cam_up = torch.nn.functional.interpolate(cam_s1, size=(224, 224), mode="bilinear", align_corners=False)
                prompts = prompt_gen(cam_up, probs_s1)
            else:
                prompts = {"point_coords": None, "point_labels": None}"""

assert old_medsam_prompt in c14_src, "old_medsam_prompt not found in Notebook 03 Cell 14"
c14_src = c14_src.replace(old_medsam_prompt, new_medsam_prompt)
nb03["cells"][14]["source"] = [l + "\n" for l in c14_src.split("\n")]
with open(nb03_path, "w", encoding="utf-8") as f:
    json.dump(nb03, f, indent=1)
print(f"Successfully updated {nb03_path} Cell 14!")
