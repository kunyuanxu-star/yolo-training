# 小数据集训练指南（~20 张图片）

## 核心原理

小数据集训练的关键矛盾是**模型容量远大于数据多样性**，容易严重过拟合。解决策略分三层：

1. **增加有效数据多样性** — 强增强让每张图在每个 epoch 看起来都不同
2. **限制模型学习能力** — 冻结 backbone、更小输入尺寸、更强正则化
3. **小心调整优化过程** — 更低学习率、更长 warmup、cosine 调度

## 推荐配置

### 命令行快速开始

```bash
python -m training_engine.train \
  --model yolov8n.pt \
  --data configs/your_dataset.yaml \
  --epochs 200 \
  --imgsz 320 \
  --batch 4 \
  --lr0 0.001 \
  --lrf 0.01 \
  --cos-lr \
  --freeze 10 \
  --close-mosaic 0 \
  --weight-decay 0.001 \
  --warmup-epochs 5 \
  --single-cls \
  --project runs/small_dataset \
  --name exp1
```

### 参数说明

| 参数 | 推荐值 | 作用 | 默认值 |
|------|--------|------|--------|
| `--epochs` | 200 | 更多 epoch，配合强增强不易过拟合 | 100 |
| `--imgsz` | 320 | 降低输入尺寸减少过拟合风险 | 640 |
| `--batch` | 2-4 | 小 batch 增加梯度噪声，有助于泛化 | 16 |
| `--lr0` | 0.001 | ★ 关键：降低 10 倍初始学习率 | 0.01 |
| `--lrf` | 0.01 | 最终学习率 = lr0 × lrf | 0.01 |
| `--cos-lr` | 开启 | 余弦退火比线性衰减更平滑 | off |
| `--freeze` | 10 | ★ 关键：冻结 backbone 前 10 层 | 无 |
| `--close-mosaic` | 0 | ★ 关键：全程保持 mosaic 不提前关闭 | 10 |
| `--weight-decay` | 0.001 | 增大权重衰减，防止过拟合 | 0.0005 |
| `--warmup-epochs` | 5 | 更长 warmup，训练初期更稳定 | 3 |
| `--single-cls` | 开启（单类时） | 简化分类头 | off |

### 两阶段训练（可选，效果更好）

**阶段 1：冻结 backbone，只训练检测头**

```bash
python -m training_engine.train \
  --model yolov8n.pt \
  --data configs/your_dataset.yaml \
  --epochs 100 \
  --imgsz 320 --batch 4 \
  --lr0 0.001 --cos-lr \
  --freeze 10 \
  --close-mosaic 0 \
  --weight-decay 0.001 \
  --project runs/small_dataset \
  --name stage1_frozen
```

**阶段 2：解冻，用更低学习率微调全网络**

```bash
python -m training_engine.train \
  --model runs/small_dataset/stage1_frozen/train/weights/best.pt \
  --data configs/your_dataset.yaml \
  --epochs 100 \
  --imgsz 320 --batch 4 \
  --lr0 0.0001 --cos-lr \
  --freeze 0 \
  --close-mosaic 0 \
  --weight-decay 0.001 \
  --project runs/small_dataset \
  --name stage2_unfrozen
```

## Web 界面配置

在 Web 界面的 ModelConfig 中设置：

| 字段 | 推荐值 |
|------|--------|
| base_model | yolov8n.pt |
| epochs | 200 |
| imgsz | 320 |
| batch | 4 |
| lr0 | 0.001 |
| lrf | 0.01 |
| cos_lr | true |
| freeze | 10 |
| close_mosaic | 0 |
| weight_decay | 0.001 |
| warmup_epochs | 5 |
| augment | true |
| single_cls | true（单类时） |

## 额外增强技巧

### 1. 通过 extra_args 开启 Ultralytics 高级增强

```bash
python -m training_engine.train \
  ... \
  --extra-args '{"mosaic": 1.0, "mixup": 0.5, "copy_paste": 0.3, "scale": 0.9, "translate": 0.2, "shear": 5.0, "hsv_h": 0.03, "hsv_s": 0.7, "hsv_v": 0.4, "fliplr": 0.5, "degrees": 10.0}'
```

### 2. 使用合成数据扩充

项目自带合成数据生成器：

```bash
python scripts/gen_synthetic_dataset.py
```

生成的数据可与真实数据混合训练。

### 3. 数据标注质量

20 张图片的场景下，标注质量比数量更重要：
- 确保每个目标框标注精确
- 避免漏标和错标
- 标注框要紧贴目标边界

## 常见问题

### 训练 loss 不下降

- 降低学习率 `--lr0 0.0005`
- 增加 warmup `--warmup-epochs 10`

### 验证集 mAP 远低于训练集 mAP（过拟合）

- 降低学习率
- 增大 weight_decay `--weight-decay 0.002`
- 保持 mosaic 开启 `--close-mosaic 0`
- 降低 imgsz `--imgsz 256`

### Loss 震荡剧烈

- 增大 batch size `--batch 8`
- 降低学习率
- 增加 warmup

## 预期效果

在 20 张高质量标注图片上，使用上述配置，单类别检测通常可以达到：
- **mAP50: 0.7–0.9**
- **mAP50-95: 0.4–0.6**

实际效果取决于：
- 目标在图片中的大小和姿态多样性
- 背景复杂度
- 标注质量
- 目标与 COCO 预训练类别的相似度
