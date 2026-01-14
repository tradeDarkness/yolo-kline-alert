# YOLO K-Line Alert Project

这是一个基于 TradingView Pine Script 指标复现与 YOLO 目标检测的加密货币交易信号与形态识别项目。

核心功能：
1. **信号检测**：在 Python 中完整复现复杂的 Pine Script 指标逻辑（均线粘合、突破、过滤器等），用于在大规模历史数据中挖掘交易信号。
2. **数据集构建**：自动将检测到的信号转换为标准化的 K 线图像和 YOLO 目标检测标签（Long/Short）。
3. **模型训练**：提供 YOLOv11 训练脚本，用于训练能够识别特定 K 线形态的模型。

---

## 📂 项目结构

```text
scripts/
├── sliding_window_signal.py  # [核心] 主程序：批量获取数据、检测信号、生成图像和标签
├── pine_signal_detector.py   # [核心] 指标逻辑：包含 MA、Oscillator、Filter 等 Pine Script 逻辑的 Python 实现
├── chart_generator.py        # [核心] 绘图模块：生成用于 YOLO 训练的标准化 K 线图
├── okx_utils.py              # [工具] OKX 数据接口：获取历史 K 线
├── prepare_yolo_data.py      # [数据] 数据集准备：划分 Train/Val 集，生成 yaml 配置
├── train_yolo.py             # [训练] YOLO 模型训练脚本
└── infer.py                  # [推理] 使用训练好的模型进行预测
```

---

## 🛠️ 安装与依赖

确保 Python 环境已安装以下依赖：

```bash
pip install pandas numpy matplotlib mplfinance requests ultralytics
```

---

## 🚀 使用指南

### 1. 信号检测与数据集生成

使用 `sliding_window_signal.py` 抓取历史数据并生成数据集。

**示例命令：**
```bash
python scripts/sliding_window_signal.py \
    --symbols "BTC-USDT-SWAP,ETH-USDT-SWAP,SOL-USDT-SWAP" \
    --limit 110000 \
    --window 60 \
    --stride 1
```

**参数说明：**
- `--symbols`: 交易对列表，逗号分隔 (默认: ETH-USDT-SWAP)。
- `--limit`: 获取 K 线数量 (默认: 110000，约 1 年 5m 数据)。
- `--window`: 图像窗口大小 (默认: 60 根 K 线)。
- `--stride`: 滑动步长 (默认: 1，建议为 1 以捕捉所有时刻)。
- `--bar`: K 线周期 (默认: 5m)。
- `--dry-run`: 仅输出信号日志，不生成图像文件。

**输出位置：**
- 图像: `data/pine_signals/images/`
- 标签: `data/pine_signals/labels/`

---

### 2. 准备 YOLO 训练数据

运行 `prepare_yolo_data.py` 将生成的原始数据划分为训练集和验证集。

```bash
python scripts/prepare_yolo_data.py
```

**功能：**
- 自动读取 `data/pine_signals` 下的图像和标签。
- 按 8:2 比例随机划分为训练集 (`train`) 和验证集 (`val`)。
- 在 `data/yolo_dataset` 下生成标准 YOLO 目录结构。
- 生成 `dataset.yaml` 配置文件。

---

### 3. 模型训练

运行 `train_yolo.py` 开始训练 YOLOv11 模型。

```bash
python scripts/train_yolo.py
```

**说明：**
- 默认加载 `yolo11n.pt` 预训练模型。
- 训练结果保存在 `runs/detect/kline_cluster_yolo11`。
- 最佳权重文件为 `weights/best.pt`。

---

### 4. 模型推理

使用训练好的模型对新图片进行检测。

```bash
python scripts/infer.py [图片路径或目录]
```

---

## ⚙️ 核心逻辑说明 (Pine Script 复现)

**复现文件**: `scripts/pine_signal_detector.py`

包含以下核心逻辑：
1. **均线系统**: 3条 SMA (20,60,120) 和 3条 EMA (20,60,120)。
2. **均线粘合 (Adhesion)**: 检测均线差值是否小于阈值。
3. **信号触发**:
   - 粘合突破 (Adhesion Breakout)
   - 密集区高低点突破 (Range Breakout)
   - 6均线顺序排列 (Strict Filter)
4. **过滤器**:
   - 动能过滤 (Momentum/Oscillator)
   - K线力度过滤 (Candle Power)
   - 均线对齐过滤 (Alignment)

**颜色样式**:
- **均线**: SMA20(黑), SMA60(蓝), SMA120(紫)
- **K线**: 灰色 (#636363)，根据 SMA100 上下关系着色逻辑已统一为灰色以保持简洁。
