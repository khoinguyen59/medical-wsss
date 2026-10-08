---
name: explainable-gnn-architect
description: "Chuyên gia kiến trúc đồ thị não và Graph AI trong Y tế (Brain GNN & Explainable Medical Graphs for fMRI / Neuroimaging): Ma trận kết nối chức năng não (Functional Connectivity), BrainGNN, GCN, giải thích mạng não (Biomarker Subgraphs) cho chẩn đoán phổ tự kỷ (ABIDE I) và đồ thị y sinh."
---

# Medical Graph AI & Brain Connectivity Architect Skill

Kỹ năng chuyên biệt phục vụ phân tích đồ thị y tế và đồ thị chức năng não (fMRI Brain Connectome) trong khuôn khổ môn học **SE374.R11** (Đặc biệt cho Đề tài số 8: Phân loại rối loạn phổ tự kỷ ABIDE I).

---

## 1. Bản Chất Đồ Thị Chức Năng Não (Brain Functional Connectivity)

1. **Từ chuỗi thời gian fMRI sang ma trận kề (Adjacency Matrix):**
   - Bộ dữ liệu ABIDE I cung cấp ảnh fMRI hoạt động não của người bình thường và người mắc rối loạn phổ tự kỷ (ASD).
   - Não được chia thành $N$ vùng quan tâm (Regions of Interest - ROIs) theo các bản đồ giải phẫu (AAL Atlas, Harvard-Oxford Atlas, Craddock 200).
   - Mỗi ROI có chuỗi tín hiệu thời gian $X_i \in \mathbb{R}^T$.
   - Tính hệ số tương quan Pearson giữa các cặp ROIs để tạo ma trận tương quan kết nối chức năng $A \in \mathbb{R}^{N \times N}$:
     $$A_{ij} = \frac{\text{Cov}(X_i, X_j)}{\sigma_i \sigma_j}$$
   - Lọc ngưỡng (Thresholding / Top-K sparsity) để tạo đồ thị não thưa không hướng.

2. **Kiến trúc mô hình GNN phân loại bệnh nhân:**
   - Sử dụng GCN, GAT hoặc BrainGNN: Đầu vào là đồ thị não của từng bệnh nhân $\rightarrow$ Dự đoán nhãn $y \in \{0, 1\}$ (Healthy Control vs Autism Spectrum Disorder).

3. **Giải thích vùng não bất thường (Biomarker Explainability):**
   - Sử dụng GNNExplainer / SubgraphX để tìm ra các cạnh liên kết và các ROIs (vùng não) có đóng góp lớn nhất vào dự đoán tự kỷ, hỗ trợ bác sĩ thần kinh xác định dấu ấn sinh học não bộ.

---

## 2. Mã Mẫu Xây Dựng & Giải Thích Đồ Thị Não

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.explain import Explainer, GNNExplainer

class BrainGNNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch=None):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        if batch is not None:
            x = global_mean_pool(x, batch)
        return self.lin(x)


def explain_brain_graph(model, data):
    """Explains which brain connections influenced the ASD prediction."""
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='graph',
            return_type='probs',
        ),
    )
    explanation = explainer(data.x, data.edge_index)
    return explanation
```
