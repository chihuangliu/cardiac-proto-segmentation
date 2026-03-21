# 研究總結報告：多尺度原型網路的可解釋心臟影像分割

**涵蓋範圍：** Stage 0–8 (Execution Plan) + Report v1–v8
**總結日期：** 2026-03-20
**資料集：** MM-WHS CT (16/2/2 patients), MRI (16/2/2 patients)
**硬體：** MacBook Pro Apple Silicon 48GB (PyTorch MPS)

---

## 一、研究動機（Research Motivation）

### 核心問題

深度學習在心臟影像分割上表現優秀，但屬於「黑盒子」系統。臨床場景（術前規劃、法規審核）要求AI不僅準確，還需可解釋。事後解釋方法（Grad-CAM、SHAP）存在根本缺陷：

- **忠實度低**（Faithfulness）：生成的熱圖只是近似，不反映模型真實計算邏輯
- **穩定度差**（Stability）：對微小輸入擾動極度敏感
- **只說「哪裡」，不說「為什麼」**：缺乏解剖學語義

### 研究方向

建立「**內建解釋性的多尺度原型分割網路**」（ProtoSegNet），沿用 ProtoPNet 的案例推理範式——「這個區域看起來像這個典型的訓練案例」，並以四個量化 XAI 指標（AP、IDS、Faithfulness、Stability）嚴格評估解釋品質。

---

## 二、各階段動機與結果

### Stage 0 — 資料管線

**動機：** 建立可重現的 MM-WHS 資料載入流程，原本預期為 3D NIfTI 體積影像。

**實際發現（重大偏差）：**
- 資料為預處理完成的 **2D 切片 (256×256) NPZ** 格式，非 3D 體積
- 嚴重類別不平衡：背景 88–94%，每個心臟結構僅 0.4–2.3%
- 原始計畫的 3D 架構（ResNet-3D、MONAI、96³ patch）全部調整為 2D

**關鍵優化：**
- `preload=True` 將資料載入從 1280 ms/batch 降至 4 ms/batch（280× 加速）

---

### Stage 1 — 2D U-Net 基線

**動機：** 在加入原型複雜性前，先驗證資料管線正確性，建立分割基準。

**結果：**

| 指標 | CT | MRI | 目標 |
|------|-----|-----|------|
| 最佳 Val Dice | 0.836 (ep20) | 0.825 (ep80) | ≥ 0.75 / 0.70 |

**重要發現：**
- CT 在 epoch 20 即達到峰值（後續 80 epoch 停滯），說明 CT 特徵空間更容易分離
- PA（肺動脈）是最難的結構（Dice 0.65–0.74），具高度幾何變異性
- MRI 訓練資料僅 CT 的一半，卻達到相近效能，代表資料品質高

**Checkpoint：** `baseline_unet_{ct,mr}.pth`

---

### Stage 2–5 — 多尺度原型網路建構

**動機：** 建立 ProtoSegNet 完整架構：

```
HierarchicalEncoder2D → PrototypeLayer（每層）→ SoftMask → Decoder → Segmentation
```

**架構規格：**

| 層級 | 步幅 | 空間大小 | 通道數 | 解剖角色 |
|------|------|---------|--------|---------|
| L1   | ×2   | 128×128 | 32     | 細紋理、邊界像素 |
| L2   | ×4   | 64×64   | 64     | 結構間邊界 |
| L3   | ×8   | 32×32   | 128    | 結構層級語境 |
| L4   | ×16  | 16×16   | 256    | 全局心臟佈局 |

**關鍵組件：**
- **PrototypeLayer**：每類別 M 個原型向量，初期採 log-cosine 相似度
- **SoftMask**：用相似度熱圖對特徵做 Hadamard 乘積，抑制背景雜訊
- **Jeffrey's Divergence 多樣性損失**：防止同類別原型坍塌（collapse）
- **三階段訓練**：Phase A（凍結原型暖機）→ Phase B（全參數訓練）→ Phase C（微調解碼器）

---

### Stage 6 — XAI 指標模組

**動機：** 建立客觀、量化的 XAI 評估體系，取代主觀視覺檢查。

**四大指標定義：**

| 指標 | 計算方式 | 意義 |
|------|---------|------|
| **AP** (Activation Precision) | `|M_k ∩ G_k| / |M_k|`，M_k 為前 5% 高激活區域 | 熱圖是否精準落在 GT 解剖區域 |
| **IDS** (Incremental Deletion) | 逐步移除高激活像素後 Dice 下降曲線的 AUC | 標記區域是否真的是決策命脈 |
| **Faithfulness** | Pearson(解釋重要性分數, Δ預測機率) | 熱圖是否反映模型真實計算 |
| **Stability** (Lipschitz) | `max ||Φ(X)−Φ(X')|| / ||X−X'||`，σ=0.05 高斯擾動 | 解釋是否在雜訊下保持穩定 |

---

### Stage 7 — 首次全面訓練與評估

**動機：** 用完整三階段訓練評估 ProtoSegNet，對比基線。

**分割結果（成功）：**

| 模型 | CT 3D Dice | MRI 3D Dice |
|------|-----------|------------|
| Baseline U-Net | 0.867 | 0.856 |
| ProtoSegNet | **0.843** | **0.805** |
| 差距 | −2.8% ✅ | −6.0% ⚠️ |

**XAI 結果（災難性失敗）：**

| 指標 | CT | 目標 | 結果 |
|------|-----|-----|------|
| AP | 0.041 | ≥ 0.70 | ❌ 差距17× |
| Faithfulness | −0.003 | ≥ 0.55 | ❌ 負相關 |
| Stability | 3.15 | ≤ 0.20 | ❌ 差距15× |

**根本原因診斷：**
- **SoftMask 太軟**：解碼器可以繞過原型遮罩，直接從未遮罩的跳接連接學習
- 原型因此對分割變得「冗餘」，其熱圖不攜帶任何空間信號
- Stability 高（3.15）確認了熱圖對輸入雜訊高度敏感，符合「均勻激活、無空間資訊」的診斷

---

### Stage 8 — XAI 修復、消融研究與視覺化

**動機：** 解決 Stage 7 的 XAI 失敗，透過消融實驗量化每個設計選擇的貢獻。

#### Phase 1：修復 XAI（Push-Pull 損失 + 相似度核心切換）

**核心發現：log-cosine 核心是 AP 失敗的根本原因**

- `log(cos_sim + 1)` 輸出有界於 `[0, 0.693]`，大多數特徵-原型對都得到中等分數，無論空間近遠
- 改用 L2 距離核心：`1 / (||z−p||²/C + 1)`，完美匹配得 1.0，隨機背景特徵得 ~0.33

**核心切換實驗結果（CT）：**

| 配置 | AP | Faithfulness | Stability |
|------|-----|-------------|-----------|
| log-cosine（Stage 7 基線）| 0.041 | −0.003 | 3.15 |
| log-cosine + push-pull (λ=0.1) | 0.013 | −0.005 | 2.44 |
| log-cosine + push-pull (λ=0.5) | 0.020 | −0.007 | 2.90 |
| **L2 + push-pull (λ=0.5)** | **0.102** | **+0.059** | 2.996 |

**成果：** AP 提升 2.5×，Faithfulness 首次轉正

#### Phase 2：消融研究

**六個變體對比（CT，50 epoch）：**

| 變體 | 描述 | Val Dice | AP | Stability | Proto Cosim |
|------|------|---------|-----|-----------|-------------|
| **Full** | L2+push-pull+div+softmask+multi-scale | 0.817 | **0.102** | 3.00 | **0.21** ✅ |
| A | 單尺度（僅 L4）| 0.810 | 0.030 | 2.46 | 0.58 |
| B | 無多樣性損失 | 0.825 | 0.130 | 14.10 ❌ | 0.64 ❌ |
| C | 無 SoftMask | 0.632 | 0.049 | 2.97 | 0.62 |
| D | 無 push-pull | 0.622 | 0.063 | 1.80 | 0.35 |
| E | log-cosine（無 push-pull）| 0.790 | 0.033 | 2.76 | 0.01 |

**五大關鍵發現：**
1. **L2 核心是 AP 主要驅動力**（Δ AP = +0.069 = +3.1× vs. log-cosine）
2. **多樣性損失是穩定性關鍵**（無 div → Stability 爆炸至 14.1，原型坍塌 cosim=0.64）
3. **SoftMask 和 push-pull 對分割缺一不可**（移除任一 → Dice 下降 >20%）
4. **多尺度提供 3.4× AP 優勢**（Full AP 0.102 vs. 單尺度 0.030）
5. **Stability 結構性下限 ≈ 2.5–3.0**（軟遮罩架構的固有限制，非超參數問題）

---

### Report v4 — 多尺度原型品質分析

**動機：** 深入理解不同尺度層的原型「學了什麼」，以及哪種層級配置最優。

**六個原型品質指標：** Purity（類別選擇性）、Utilization（死亡原型率）、Spatial Compactness（空間集中性）、Dice Sensitivity（因果重要性）、Level Dominance（像素主導比例）、Per-level AP

**M4（L1-L4） Post-Hoc 分析：**

| 層級 | Purity | 像素主導比例 | 角色 |
|------|--------|------------|------|
| L1 | 0.050 | 34% | 低語義、主導像素，注入雜訊 |
| L2 | 0.184 | 44% | 低語義、主導像素，注入雜訊 |
| L3 | 0.639 | 17% | 有意義 |
| L4 | 0.824 | **4%** | 最高語義，卻幾乎無法主導像素 |

**Purity 悖論：** L4 原型 purity 最高（0.824）→ 卻只主導 4.3% 像素，因為 L1/L2 的寬泛激活在 winner-takes-all 聚合中將 L4 信號壓制。

**三種配置比較（M4/M2/M1）：**

| 模型 | 層級 | 3D Dice | 整體 Purity | L4 主導比例 |
|------|------|---------|-----------|-----------|
| M4 | L1+L2+L3+L4 | 0.841 | 0.334 | 4.3% |
| M1 | L4 only | 0.852 | 0.499 | 100% |
| **M2** | **L3+L4** | **0.872** | **0.733** | **49.1%** |

**結論：M2（L3+L4）是最優配置**
- 分割最佳（+3.2% vs M4）
- Purity 最高（0.733）
- 36% 原型有因果重要性（M4 僅 9%）
- L3/L4 各主導 ~50% 像素，證明互補而非一強獨大

**設計原則：包含語義層（L3, L4），排除紋理層（L1, L2）**

---

### Report v5 — 學習式層級注意力

**動機：** 能否用學習式注意力機制「自動發現」L1/L2 無用的事實，而非需要手動消融？

**架構：LevelAttentionModule**
- 用 encoder 各層的 GlobalAvgPool 特徵，透過 MLP 預測各層的 softmax 權重 w
- 用加權混合取代 winner-takes-all max 聚合
- 僅增加 ~31K 參數（< 0.1%）

**實驗 1（λ_ent = 0.02，含 entropy 正則化）：**
- 結果：w 幾乎維持均勻（L1~L4 各 0.25）
- 原因：entropy 梯度在均勻點為零，無法驅動偏移

**實驗 2（λ_ent = 0，無正則化）：**

| Epoch | w_L1 | w_L2 | w_L3 | w_L4 |
|-------|------|------|------|------|
| 35    | 0.006| 0.010| 0.340| **0.644** |
| 60    | 0.000| 0.000| 0.075| **0.925** |
| 100   | 0.000| 0.000| 0.060| **0.940** |

**RQ5 回答：確認自動發現 L1/L2 無用**
- 解凍後 5 個 epoch 內，L1/L2 → 0，L4 → 0.94
- 完全復現 v4 手動消融的發現，但不需要任何先驗知識

**但注意力機制未能取代 M2 的效果：**
- M4-attn(λ=0)：3D Dice 0.842 vs M2：0.872（差距 3.1%）
- 原因：學習式注意力只調整 soft mask 貢獻，L1/L2 的梯度仍流通 encoder

---

### Report v6 — 縮小注意力-M2 差距的嘗試

**動機：** 三個實驗嘗試縮小 M4-attn 與 M2 之間 3.1% 的 Dice 差距。

**Exp A — 注意力加權原型損失（RQ6）：**
- 以各層注意力權重縮放其 prototype 損失（`.detach()` 防止梯度回流至注意力 MLP）
- 結果：Purity +0.160，AP +0.110，但 L2 自我增強迴圈（w_L2 鎖定在 ~10%）

**Exp B — 漸進式層修剪（RQ7）：**
- 自動偵測 w_l < 0.05 → 修剪該層
- L1/L2 確實自動被修剪，但未能達到 M2 效能
- 三大結構性問題：解碼器已與 4 層共同適應、剩餘訓練時間不足、Encoder 已被 4 層塑形

**Exp D — 兩階段暖啟動（RQ9）：**
- Stage 1（發現）：訓練 M4-attn，找出無用層
- Stage 2（訓練）：以 Stage 1 encoder 初始化新 M2 模型
- 結果：因 L2 迴圈，Stage 1 發現 L2+L4 而非 L3+L4，L2 污染攜入 Stage 2

**統一結論：** 所有基於 M4 的修改均無法超越冷啟動 M2。L2 encoder 污染是核心問題，無法被事後補救。

---

### Report v7 — 自動化層選擇的極限

**動機：** 能否在不需要人工指定 L3+L4 的情況下，自動化地選出最優層？

**Stage 29 — 暖啟動 L3+L4（手動指定）：**
- 忽略 Stage 1 的噪聲發現，直接使用 L3+L4
- 結果：3D Dice 0.8656，Effective Purity 0.649（vs v6 warmstart 的 0.267）

**Stage 30 — 修復 LevelAttentionModule（feature detach + temperature annealing）：**
- 目標：讓 Stage 1 穩定發現 L3+L4
- 結果：機制正確，但收斂至 L2+L3（非 L3+L4）
- **根本原因：目標函數不匹配**
  - 分割損失（Dice/CE）→ 傾向 L2（空間解析度高，有利解碼器重建）
  - 原型解釋性（purity, AP）→ 傾向 L4（語義深度高）
  - 注意力模組優化的是分割損失，不是解釋性目標

**Stage 31 — 事後消融作為探索工具：**
- 在 M4 上 zero-out 各層組合，評估 15 個非空子集
- 結果：Pareto 前沿有 5 個子集，無法自動選擇；且 {L3,L4} 的 post-hoc Dice（0.695）遠低於專門訓練的 M2（0.866），存在 co-adaptation bias

**核心結論：**
- 訓練時自動選層（注意力）和推論時自動選層（消融）均告失敗
- **人工指定 L3+L4 是目前唯一可靠且有實驗支撐的選擇**

---

### Report v8 — 兩階段管線與解剖局部約束

**動機：** 能否用純資料驅動的無閾值方法自動選出 L3+L4？能否用空間約束改善 purity？

**Stage 32 — Plain M4 + Max-Gap 過濾器（RQ13）：**
- 計算 plain M4（無注意力）各層 purity，找最大 gap 自動切割

| 過渡 | Gap |
|------|-----|
| L1→L2 | 0.111 |
| **L2→L3** | **0.418 ← 最大** |
| L3→L4 | 0.076 |

**RQ13 通過：** Max-gap 過濾器自動無誤地選出 L3+L4，無需任何硬編碼閾值。

**Stage 34 — M2(L3+L4) + Anatomical Locality Constraint（ALC，RQ14）：**
- 每個原型的激活重心應靠近其類別的解剖學位置（Soft-argmax 可微分實現）
- 結果：ALC 損失值整個訓練過程幾乎平坦，centroid deviation 無改善（37px）
- **根本原因：原型投影後熱圖呈尖銳峰值，softmax 飽和，梯度消失**

| 模型 | 3D Dice | Eff. Purity | Centroid Dev. |
|------|---------|------------|---------------|
| Stage 29（無 ALC）| 0.8656 | 0.649 | — |
| Stage 34（ALC L3+L4）| 0.8478 | 0.661 | 37px |
| Stage 34b（ALC 僅 L3）| 0.8628 | 0.593 | 37px |

**RQ14 未通過：** ALC 無法改善解剖局部性，公式需根本修改（改用 ReLU 正規化或低溫 softmax）。

**最終研究成果（兩階段管線）：**
```
階段 1：訓練 Plain M4 → Max-Gap Filter → 自動選出 L3+L4
    ↓
階段 2：以 Stage 1 encoder 暖啟動 M2(L3+L4) → 訓練 100 epochs
    ↓
結果：Dice=0.8628, Eff.Purity=0.649（冷啟動 M2：0.8722）
```

---

## 三、全研究數字總覽

| 模型 | CT 3D Dice | AP | IDS | Faithfulness | Stability | Proto Cosim |
|------|-----------|-----|-----|-------------|-----------|-------------|
| Baseline U-Net | 0.867 | — | — | — | — | — |
| ProtoSegNet cosine（Stage 7）| 0.843 | 0.041 | 0.047 | −0.003 | 3.15 | — |
| ProtoSegNet L2（Stage 8 best）| 0.843 | **0.102** | **0.007** | **+0.059** | 3.00 | **0.21** |
| M4-attn λ=0（v5）| 0.842 | 0.085 | — | — | — | — |
| **M2（v4 cold-start）** | **0.872** | 0.189 (L4) | — | — | — | — |
| Two-phase M2 warm-start（v8）| 0.863 | — | — | — | — | — |

---

## 四、撰寫報告的方向建議

基於上述研究軌跡，以下提供三種不同的組合方向，各有其適合的學術發表情境。

---

### 組合 A：系統性論文（Conference Paper，方法貢獻型）

**定位：** 完整呈現 ProtoSegNet 方法，強調架構設計與 XAI 指標評估。適合 MICCAI、MIDL、ISBI 等醫學影像頂會。

**建議章節結構：**

1. **Introduction** — 黑盒子問題、事後解釋的侷限、原型網路的優勢、本研究貢獻
2. **Related Work** — ProtoPNet、MProtoNet、ProtoSeg、HierViT；XAI 指標（AP/IDS/Faithfulness/Stability）
3. **Dataset** — MM-WHS、2D 切片格式、類別不平衡、資料載入優化（preload）
4. **Method (ProtoSegNet)** — 多尺度 encoder、L2 相似度核心、SoftMask、Push-Pull 損失、Jeffrey's Divergence 多樣性損失、三階段訓練
5. **Results**
   - 基線比較（Stage 1 vs Stage 8 ProtoSegNet）
   - XAI 指標：L2 vs cosine 核心的對比（Stage 8 Table 1）
   - 消融研究 6 變體（Stage 8 Table 2）
6. **Discussion** — Stability 結構性下限、AP 天花板、PA/LA 困難結構
7. **Conclusion** — 貢獻摘要、future work

**使用核心內容：** Report v1（完整方法描述）+ Execution Plan Stage 8（消融結果）

**優點：** 結構完整，有清晰的 research question → method → result 邏輯。

**注意：** 需誠實說明 Stability/AP 未達原始目標，但有清楚的技術解釋（soft-mask 結構性限制）。

---

### 組合 B：深度分析論文（Workshop / Short Paper，分析貢獻型）

**定位：** 聚焦在「原型品質」這個新研究問題，展示 purity 悖論與最優層配置發現。適合 CVPR/MICCAI Workshop、XAI 相關 venue。

**建議章節結構：**

1. **Introduction** — 多尺度原型分割的既有問題：哪些層有用？如何衡量「原型品質」？
2. **Prototype Quality Framework** — 六個新指標（v4 §3）：Purity、Utilization、Compactness、Dice Sensitivity、Level Dominance、Per-level AP
3. **M4 Post-hoc Analysis** — Purity 悖論：L4 最純卻最不主導（v4 §4）
4. **Ablation: M4 vs M2 vs M1** — M2 最優的量化確認（v4 §5）
5. **Validation: Learned Attention Confirms the Hierarchy** — 注意力自動發現 L4=0.94 的強化實驗（v5）
6. **Conclusion** — 設計原則：保留語義層（L3/L4），去除紋理層（L1/L2）

**使用核心內容：** Report v4（主體）+ Report v5（強化驗證）

**優點：** 提出「原型品質指標」作為新貢獻，角度新穎，資料充分，結論清晰。

**注意：** 要解釋為何使用 CT-only 資料，MR 驗證未完成需列為 limitation。

---

### 組合 C：過程探索論文（Full Paper，負面結果型）

**定位：** 呈現自動化層選擇的完整研究弧線，包括所有失敗的嘗試與學到的原則。適合 NeurIPS 負面結果 Track、MLSys、或深度學習方法論期刊。

**建議章節結構：**

1. **Introduction** — 多尺度原型的層選擇問題；自動化的必要性
2. **Baseline: M2 是最優配置** — v4 結論，作為後續實驗的黃金標準
3. **嘗試 1：學習式注意力** — 成功自動發現 L4=0.94，但無法取代 M2（v5）
4. **嘗試 2：注意力加權損失** — L2 feedback loop 阻止完全收斂（v6-A）
5. **嘗試 3：漸進式修剪** — Decoder co-adaptation 問題（v6-B）
6. **嘗試 4：兩階段暖啟動** — L2 encoder 污染攜入 Stage 2（v6-D/v7）
7. **嘗試 5：修復注意力模組** — 機制正確，但目標函數不匹配（v7-Stage 30）
8. **解決方案：Max-Gap 過濾器** — v8 的正面結果，簡單有效
9. **Anatomical Locality Constraint** — 軟 argmax 飽和的教訓（v8）
10. **Conclusions & Principles** — 正反面發現的系統化總結

**使用核心內容：** Report v4 → v5 → v6 → v7 → v8 的完整研究弧線

**優點：** 研究過程透明，負面結果有重要學術價值，分析深度最高。

**注意：** 敘事需要有「統一的研究問題」貫穿，避免看起來像雜亂的實驗日誌。建議統一問題設定為：「如何在不需要人工介入的情況下，自動選擇最優的原型層配置？」

---

### 各組合適用情境速覽

| 組合 | 定位 | 頁數 | 適合 Venue | 核心貢獻 |
|------|------|------|----------|---------|
| A | 系統性方法論文 | 8–10 | MICCAI, MIDL, ISBI | ProtoSegNet 架構 + XAI 指標評估 |
| B | 原型品質分析 | 4–6 | Workshop, XAI venue | 六個 prototype quality 指標 + purity 悖論 |
| C | 自動化層選擇探索 | 12–15 | NeurIPS, MLSys, 期刊 | 完整研究弧線 + 設計原則提煉 |

---

## 五、所有組合共用的「必寫」部分

無論選擇哪種組合，以下內容幾乎必然需要呈現：

1. **L2 vs. log-cosine 核心的比較**（Stage 8 Table 1）——這是全研究最乾淨的單變量對比
2. **Jeffrey's Divergence 多樣性損失的必要性**（消融 Variant B）——移除後 Stability 爆炸至 14.1，非常有說服力
3. **Soft-mask 與 push-pull 的協同效果**（消融 Variant C/D）——各自移除 Dice 均下降 >20%
4. **Purity 悖論**（v4）——L4 purity 最高卻像素主導最低，直觀且反直覺
5. **Stability 結構性下限的解釋**——是架構選擇的後果，不是訓練問題

---

## 六、尚未完成的工作（Limitations / Future Work）

| 項目 | 狀態 | 影響 |
|------|------|------|
| MR L2 retrain | 未完成 | MR XAI 指標仍為 cosine 版本，無法完整比較 |
| MR max-gap 驗證（Stage 35）| 未完成 | L3+L4 最優性未在 MR 確認 |
| ALC 正確公式（ReLU normalisation）| 未嘗試 | 解剖局部約束仍是 open problem |
| 硬遮罩架構測試 | 未完成 | Stability < 1.0 的唯一潛在解法 |
| 測試集僅 2 患者/模態 | 設計限制 | 所有 XAI 指標應視為 case study，非統計估計 |
