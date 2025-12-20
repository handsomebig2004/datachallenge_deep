# datachallenge_deep

transformer最强


下面按“类型”把**已经用过**的模型和**还可以用**的模型列出来（只列结构名，不展开细节）。

---

## 1) U-Net 系列（Encoder–Decoder / Skip Connection）

### 你已用

* **SegNet / segnet_gen 0.62125**
* **Attention-U-Net  0.5137568332816138**
* **ResNet34-U-Net 0.4858865707148295**
* **U-Net（keep272）0.4850**

### 仍可用

* **ResNet18/50-U-Net**
* **UNet++（Nested U-Net）**
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
* **Mask2Former（更强但更复杂） 0.6441724128817745**
* **UPerNet + Swin / ViT backbone  0.6402961815634437**
* **DPT（Dense Prediction Transformer）0.6234547207117457**
* **Swin-UNet / Swin-Transformer segmentation  0.5850994834704769**

### 仍可用


---

## 4) Pyramid / FPN 多尺度融合（CNN 语义分割头）

### 你已用

* **UPerNet（PSP + FPN）0.5387714111687977**

### 仍可用

* **PSPNet**
* **FPN-Seg**
* **HRNet + OCR（高分辨率）**3
* **PAN（Path Aggregation Network）**

---

## 5) “边界/形状友好”的结构（适合细长界面）

### 你已用

* （暂无）

### 仍可用

* **Boundary Head（分割 + 边界分支）**
* **Distance Transform 回归 + 分割**
* **Gated-SCNN（边界引导）**4

---

### 👉 结论（不拐弯）：

> **你现在继续“堆更复杂 Transformer 结构”，收益已经开始下降了。**

原因不是模型不强，而是：

* 数据量有限（几千 patch）
* 标签噪声 + 细长结构
* 评价是 **IoU（对边界极其敏感）**

在这种任务里，**SegFormer 已经吃满了“Transformer 红利”**。

---

## 🎯 现在最有可能再涨分的 3 条路（按性价比排序）

### 🥇 路线 A（最推荐）：**SegFormer + 强化训练策略**

这是**最可能把 0.649 → 0.67+** 的方式。

你可以加：

1. **Dice / Tversky / Lovász-Softmax loss（替代纯 CE）**
2. **TTA（左右翻转 + 轻微 scale）**
3. **连通域 / morphology 后处理（修断裂）**

👉 不换模型，只“榨干 SegFormer”

---

### 🥈 路线 B：**Mask2Former（现在可跑，但要“比赛版”配置）**

如果你坚持用它，**一定要注意**：

* batch size = 1 或 2
* 冻结 backbone 前几层
* 更长 warmup
* 不然很容易过拟合 well1–5

👉 **上限高，但不稳**

---

### 🥉 路线 C：**UPerNet + Swin + 边界辅助**

单纯 UPerNet+Swin 已经证明不够，你需要：

* 加 **Boundary Head**
* 或 Distance Transform loss

👉 复杂度高，回报不确定

---

## 🚀 我建议你现在这样走（最理性）

> **把 SegFormer 当主模型冲榜**
> Mask2Former 作为“你已经会了的备选方案”

---

