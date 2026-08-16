# Datasets

每个子目录是一个单类别 YOLO 检测数据集，目录结构与 Ultralytics
`data.yaml` 约定一致（`train/images`、`train/labels`、`val/images`、
`val/labels`）。

| 类别 | 训练图 | 验证图 | 来源 |
| --- | --- | --- | --- |
| apple | 160 | 40 | 仓库既有 Roboflow 数据集 |
| banana | 351 | 64 | Open Images v6 (`/m/09qck`) |
| paper_cup | 171 | 30 | 仓库既有 Roboflow 数据集 |
| orange | 469 | 64 | Open Images v6 (`/m/0cyhj_`) |
| doll | 500 | 64 | Open Images v6 (`/m/0167gd`) |
| plastic_bottle | 523 | 92 | 仓库既有 Roboflow 数据集 |
| book | 500 | 64 | Open Images v6 (`/m/0bt_c3`) |
| chair | 500 | 64 | Open Images v6 (`/m/01mzpv`) |
| cup | 425 | 64 | Open Images v6 (`/m/02p5f1q`) |
| computer_keyboard | 500 | 64 | Open Images v6 (`/m/01m2v`) |
| laptop | 500 | 64 | Open Images v6 (`/m/01c648`) |

Open Images 图片从公共 S3 桶
`open-images-dataset` 匿名下载，仅保留 `IsDepiction=0`、
`IsGroupOf=0`、`IsOccluded=0`、`IsTruncated=0`，且框面积在
`[0.03, 0.85]` 之间的真实照片标注，并转为 YOLO 格式。
