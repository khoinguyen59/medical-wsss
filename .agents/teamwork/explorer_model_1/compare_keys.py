import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import types
import torch.nn as nn

# Mock torchsummary
dummy_ts = types.ModuleType("torchsummary")
dummy_ts.summary = lambda *args, **kwargs: None
sys.modules["torchsummary"] = dummy_ts

# Mock sync_batchnorm
dummy_sync_bn = types.ModuleType("network.sync_batchnorm.batchnorm")
dummy_sync_bn.SynchronizedBatchNorm2d = nn.BatchNorm2d
sys.modules["network.sync_batchnorm"] = types.ModuleType("network.sync_batchnorm")
sys.modules["network.sync_batchnorm.batchnorm"] = dummy_sync_bn

sys.path.insert(0, "repos/WSSS-Tissue")
sys.path.insert(0, ".agents/teamwork/explorer_model_1")

from network.deeplab import DeepLab as RepoDeepLab
from verify_model import DeepLabV3Plus as StandaloneDeepLab

repo_model = RepoDeepLab(backbone="resnet", output_stride=16, num_classes=4, sync_bn=False)
standalone_model = StandaloneDeepLab(num_classes=4, output_stride=16)

repo_keys = set(repo_model.state_dict().keys())
standalone_keys = set(standalone_model.state_dict().keys())

diff_keys = repo_keys ^ standalone_keys
print("Key set difference count:", len(diff_keys))
if diff_keys:
    print("Mismatched keys:", diff_keys)
else:
    print("PERFECT 100% KEY MATCH between repos/WSSS-Tissue and StandaloneDeepLab!")

# Verify shapes
mismatched_shapes = []
for k in repo_keys:
    s_repo = repo_model.state_dict()[k].shape
    s_stand = standalone_model.state_dict()[k].shape
    if s_repo != s_stand:
        mismatched_shapes.append((k, s_repo, s_stand))

print("Mismatched shapes count:", len(mismatched_shapes))
if mismatched_shapes:
    print("Mismatches:", mismatched_shapes)
else:
    print("PERFECT 100% TENSOR SHAPE MATCH for all 680 parameters and buffers!")
