# Dispatch Assignment: team2_fixer_colab

## Objective
Update scripts/generate_colab_variants.py and regenerate all 4 GPU-specific Colab notebooks:
1. Notebooks must verify and download both official BCSS checkpoints:
   - stage1_checkpoint_trained_on_bcss.pth (Google Drive ID 118y3Q7i6lPOrw8OIFqt3tlP1PUM-i1Xc)
   - stage2_checkpoint_trained_on_bcss.pth (Google Drive ID 1cXm29ko8UevbaeW6YreSg-vuNaQiN_je)
2. Notebooks must instantiate both model (Stage 2 DeepLabV3+ ResNet-101) and model_stage1 (WSSSTissueStage1NetCAM ResNet38d), evaluate all 5 methods on real BCSS-WSSS data, export ablation tables and render comparison grids.
3. Regenerate all 4 GPU variants:
   - notebooks/SE374_Colab_WSSS_T4_Sanity150.ipynb
   - notebooks/SE374_Colab_WSSS_T4_Full4986.ipynb
   - notebooks/SE374_Colab_WSSS_L4_Full4986.ipynb
   - notebooks/SE374_Colab_WSSS_A100_Full4986.ipynb
   via cmd /c "python scripts/generate_colab_variants.py".
4. Ensure notebooks/SE374_Colab_WSSS_Pipeline.ipynb and tests/test_colab_notebook.py pass 100%.
5. MANDATORY INTEGRITY WARNING:
   DO NOT CHEAT. All implementations must be genuine.
6. Strict Criteria:
   - 100% ASCII 0x2D hyphens only (zero em-dashes/en-dashes).
   - All shell executions via cmd /c "...".
   - Zero scratch scripts.
   - Run tests: cmd /c "python -m pytest tests/test_colab_notebook.py -v" and report results.
   - Write handoff.md and progress.md in your working directory and notify orchestrator when done.
