# datachallenge_deep

transformer最强
下面按“类型”把**已经用过**的模型和**还可以用**的模型列出来（只列结构名，不展开细节）。

---

## 1) U-Net 系列（Encoder–Decoder / Skip Connection）

### 你已用

* **U-Net（keep272）0.4850**
* **ResNet34-U-Net 0.4858865707148295**

### 仍可用

* **ResNet18/50-U-Net**
* **UNet++（Nested U-Net）**
* **Attention U-Net**
* **U-Net 3+**
* **R2U-Net / R2AttU-Net（递归残差）**
* **BCDU-Net**
* **DenseUNet**
* **U-Net + ASPP（在瓶颈加 ASPP）**

---

## 2) DeepLab / ASPP 多尺度语义分割（Atrous）

### 你已用

* **DeepLabV3（ResNet50 backbone）0.46073617997736044**

### 仍可用

* **DeepLabV3+**（比 V3 更强的 decoder）
* **DeepLabV3-MobileNetV3**（更快）
* **LR-ASPP（轻量 ASPP）**

---

## 3) Transformer / ViT 系列语义分割

### 你已用

* **SegFormer 0.649377**

### 仍可用

* **Swin-UNet / Swin-Transformer segmentation**
* **Mask2Former（更强但更复杂）**
* **UPerNet + Swin / ViT backbone**
* **DPT（Dense Prediction Transformer）**

---

## 4) Pyramid / FPN 多尺度融合（CNN 语义分割头）

### 你已用

* **UPerNet（PSP + FPN）0.5387714111687977**

### 仍可用

* **PSPNet**
* **FPN-Seg**
* **HRNet + OCR（高分辨率）**
* **PAN（Path Aggregation Network）**

---

## 5) “边界/形状友好”的结构（适合细长界面）

### 你已用

* （暂无）

### 仍可用

* **Boundary Head（分割 + 边界分支）**
* **Distance Transform 回归 + 分割**
* **Gated-SCNN（边界引导）**

---

如果你要我下一步直接落地，我建议你从**“还可用”里最值得做、且代码不会爆炸**的两类选一个：

* **UNet++**（最贴合你任务的细长结构）
* **DeepLabV3+**（比 V3 更强，改动也不算太大）
