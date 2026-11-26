# YOLO Platform V2.0 - 开发说明

本文档说明代码实现状态、运行方式与遗留事项。产品功能文档见 [README.md](README.md) 与 [docs/zh/](docs/zh/)。

## 目录

- [本地运行](#本地运行)
- [上传项目到 GitHub（一步步）](#上传项目到-github一步步)
- [AutoDL 云 GPU 训练（一步步）](#autodl-云-gpu-训练一步步)
- [已实现功能](#已实现功能对照设计文档)
- [已知限制与遗留事项](#已知限制与遗留事项待后续完成)
- [测试](#测试)
- [目录结构](#目录结构)

---

## 本地运行

```bash
# 本机（Windows + venv 已就绪）
venv\Scripts\activate
python main.py
```

- 首次启动自动生成 `config/config.yaml`、`workspace/` 目录与默认管理员 **admin / admin123**（登录后可在设置中修改）。
- 首次使用 AI 标注/训练时会自动下载预训练模型（如 `yolo11n.pt`，来源为 ultralytics 官方 GitHub Releases）。

---

## 上传项目到 GitHub（一步步）

> 目的：让 AutoDL 云服务器能直接 `git clone` 你的代码。
> 注意：当前本地仓库的远程地址是原作者的仓库（你没有推送权限），需要改成**你自己的仓库**。
> 模型文件（*.pt）和数据集（workspace/）已在 `.gitignore` 中排除，不会上传。

### 第 1 步：注册 GitHub 账号（如已有可跳过）

打开 https://github.com → Sign up → 用邮箱注册并验证。

### 第 2 步：在 GitHub 网站上新建空仓库

1. 登录后点击右上角 **+** → **New repository**
2. Repository name 填 `YOLO_Platform`（或任意名字）
3. 选 **Private（私有）** 或 Public 均可
4. **不要**勾选 "Add a README file"（保持空仓库）
5. 点 **Create repository**
6. 创建完成后页面会显示仓库地址，形如：
   `https://github.com/zaijienantao-jiejie/YOLO_Platform.git` —— 记下这行

### 第 3 步：在本地提交代码并推送

打开终端（在项目目录 `D:\github项目\YOLO_Platform` 下），依次执行：

```bash
# 1) 把全部代码加入暂存区
git add .

# 2) 提交（首次提交）
git commit -m "实现 YOLO Platform V2.0 完整功能"

# 3) 把远程地址从原作者仓库换成你自己的仓库（把 zaijienantao-jiejie 换成第 2 步的名字）
git remote add origin https://github.com/zaijienantao-jiejie/YOLO_Platform.git

# 4) 推送
git push -u origin main
```

推送时如果弹出登录窗口，用浏览器授权登录 GitHub 即可（或提前安装 GitHub Desktop 登录）。

### 第 4 步：验证

刷新 GitHub 仓库页面，能看到 `main.py`、`gui/`、`services/` 等文件就成功了。

### 以后更新代码

```bash
git add .
git commit -m "更新说明"
git push
```

---

## AutoDL 云 GPU 训练（一步步）

> 本地是 AMD 显卡（无 CUDA），训练慢；在 AutoDL 租 NVIDIA GPU 训练，按小时计费，用完关机释放。

### 第 1 步：注册并充值

1. 打开 https://www.autodl.com → 注册登录
2. 顶部"充值"，充 10~50 元足够跑好几次实验

### 第 2 步：租用实例

1. 点"算力市场"→"租用新实例"
2. 选择：
   - **GPU**：RTX 3090（约 1 元/小时，24GB 显存，性价比首选）或 RTX 4090（更快）
   - **镜像**：基础镜像里选 **PyTorch 2.x + Python 3.10 + CUDA 12.x** 的版本
   - 地区选离你近的
3. 点"立即创建"，等待开机（1~2 分钟）
4. 开机后点 **JupyterLab** 打开网页终端（不需要远程桌面，全程用命令）

### 第 3 步：上传代码和数据集到 AutoDL

**代码**：在 AutoDL 的 JupyterLab 终端里执行：

```bash
git clone https://github.com/zaijienantao-jiejie/YOLO_Platform.git
```

**数据集**（两种方式任选）：

- **方式 A（推荐，大文件）**：AutoDL 网页左侧"我的网盘"（免费有额度）→ 上传本地
  `workspace/datasets/药材分类导入测试/` 整个文件夹 → 上传完后在实例里挂载网盘，
  把文件夹移动到项目里：
  ```bash
  mv ~/autodl-tmp/药材分类导入测试 ~/YOLO_Platform/workspace/datasets/
  ```
- **方式 B（文件不大时）**：直接在 JupyterLab 左侧文件树把本地文件夹拖拽上传到
  `~/YOLO_Platform/workspace/datasets/`

> 平台数据集 = 一个包含 `dataset.json`、`images/`、`labels/` 的文件夹，整个文件夹传过去即可，
> 平台能直接识别。也可以把原始数据集传上去后，用平台代码重新导入。

### 第 4 步：安装环境

在 AutoDL 终端（JupyterLab 里）：

```bash
cd ~/YOLO_Platform

# 创建虚拟环境
conda create -n yolo python=3.10 -y
conda activate yolo

# 安装 GPU 版 PyTorch（CUDA 12.4）
# 提示：AutoDL 镜像若已自带 PyTorch（python -c "import torch" 不报错），可跳过这行
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 精简安装：命令行训练只需要 ultralytics（自动带上 numpy/opencv/pyyaml 等）
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple

# 完整安装（含界面库 PySide6 等；服务器上不跑界面时非必需）
# pip install -r requirements-gpu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 第 5 步：确认数据集与 GPU 正常

```bash
python train_cli.py --list-datasets     # 应看到你的数据集和图像数
python -c "import torch; print(torch.cuda.get_device_name(0))"   # 应显示 GPU 型号
```

### 第 6 步：开始训练（命令行，无需界面）

```bash
# 药材分类（示例）—— 93 类 9000 图，GPU 上每个 epoch 约几十秒
python train_cli.py --dataset 药材分类导入测试 --model yolo11n-cls \
    --epochs 100 --batch 128 --imgsz 224 --device 0

# 猫狗检测（示例）
python train_cli.py --dataset cats_dogs导入测试 --model yolo11s \
    --epochs 200 --batch 32 --imgsz 640 --device 0

# 中断后断点续训
python train_cli.py --dataset 药材分类导入测试 --model yolo11n-cls --resume --device 0
```

训练过程中会实时打印每个 epoch 的损失/精度。训练完输出：

``` 
训练完成！
  最佳模型: ~/YOLO_Platform/workspace/models/train_yolo11n-cls/weights/best.pt
```

### 第 7 步：把训练结果拿回本地

1. 在 AutoDL 的 JupyterLab 文件树里找到 `workspace/models/train_xxx/weights/best.pt`，右键下载
   （或先把整个 `train_xxx` 文件夹上传到"我的网盘"，再从网盘下载）
2. 本地放入 `D:\github项目\YOLO_Platform\workspace\models\` 下任意位置
3. 打开本地软件 → "📊 评估"页选择 best.pt → 评估；或 "🌐 部署"页 → 推理 / API / 导出

### 第 8 步：用完关机（重要！）

AutoDL 按小时计费，训练完务必到控制台点 **"关机"**（关机后不再计费，数据保留）；
长期不用可点"释放实例"（数据会清空，记得先把模型下载回本地）。

### 常见问题

| 问题 | 解决 |
|------|------|
| pip 下载慢 | 命令加 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |
| `ModuleNotFoundError: No module named 'PySide6'` | 命令行训练不需要 PySide6。更新代码到最新版（`git pull`）后重试；只需安装 ultralytics 即可 |
| 预训练模型下载慢 | 把本地 `workspace/models/pretrained/*.pt` 一起传到 AutoDL 的对应目录，训练时自动使用本地模型 |
| git clone 失败 | 确认仓库是 Private 时需要先配置 token 或用 SSH 地址 |
| 想跑 GUI | AutoDL 可租"桌面版镜像"或使用 JupyterLab 的图形界面功能；训练用 `train_cli.py` 即可 |
| 显存不足(OOM) | 减小 `--batch`（如 16/8），或增大 `--imgsz` 改为减小（如 160） |

---

## 已实现功能（对照设计文档）

| 模块 | 状态 | 说明 |
|------|------|------|
| 数据集管理 | ✅ | 5 种任务创建；YOLO/COCO/VOC/文件夹导入（支持子目录与 train/val 分目录布局）；YOLO/COCO/VOC 导出；划分；快照/恢复；复制/重命名/删除 |
| 交互式标注 | ✅ | 矩形框（可拖动/四角缩放）、多边形、关键点（含骨架连线）、旋转框（旋转手柄）、分类标签；撤销/重做（Ctrl+Z/Y）、保存（Ctrl+S）、下一张（Tab） |
| AI 辅助标注 | ✅ | 全 5 任务；预训练模型选择（YOLOv8/11/12）；置信度/IoU/关键点阈值；类别映射（COCO 自动匹配同名类别）；仅标注未标注图像；AI 标记 🤖，人工修改后自动清除 |
| 数据验证 | ✅ | 图像损坏、标注越界、类别 ID、关键点数量、多边形、旋转角度检查，生成验证报告 |
| 离线数据增强 | ✅ | 水平/垂直翻转、旋转 90°、HSV、亮度对比度、高斯噪声、Mosaic（检测）；姿态翻转自动左右关键点交换 |
| 模型训练 | ✅ | YOLOv8/11/12 全系列 n/s/m/l/x；实时指标（损失/mAP/学习率曲线）；AMP；早停；断点续训；停止按钮；`train_cli.py` 命令行入口（云 GPU） |
| 模型评估 | ✅ | 验证集指标、混淆矩阵/PR 曲线等图表、FPS 性能测试、模型大小 |
| 模型导出 | ✅ | ONNX/OpenVINO/TFLite/TensorRT/TorchScript/CoreML/Paddle；FP16/INT8/ONNX 简化/动态尺寸 |
| 本地推理 | ✅ | 图像推理（结果图）、视频/摄像头实时推理（FPS 显示） |
| API 服务 | ✅ | FastAPI：/predict、/predict/batch、/predict/stream、/health、/info、/metrics |
| GUI | ✅ | 无边框窗口、折叠侧边栏、6 种主题、状态栏设备信息 |
| 系统工具 | ✅ | 系统诊断（依赖/硬件/GPU 检测）、缓存清理、日志查看器（级别/关键词/导出） |
| 认证 | ✅ | 登录/访客模式、用户管理（增删/改密/角色）、会话超时、pbkdf2:sha256 密码存储 |

## 已知限制与遗留事项（留待后续完成）

1. **多 GPU（DDP）训练**：训练接口未暴露多卡参数（ultralytics 支持 device=0,1，后续可在训练面板加设备多选）。
2. **标注质量检查**：审核工作流中的重叠检测、类别一致性自动检查未实现（验证报告已覆盖基础项）。
3. **多人协作标注**：设计文档中标注为未开发项，保持未实现。
4. **单元测试覆盖**：仅有冒烟测试（`tests/test_smoke.py`、`tests/test_gui_threading.py`），GUI 交互未完全自动化。
5. **Paddle/CoreML 导出**：需额外依赖（paddlepaddle / macOS），失败时界面有明确提示。
6. 设计文档中的 "YOLO26" 为虚构系列，实际支持 YOLOv8/11/12。

## 测试

```bash
venv\Scripts\python tests\test_smoke.py             # 非 GUI 冒烟测试
venv\Scripts\python tests\test_gui_threading.py    # GUI 线程化离屏测试
```

## 目录结构

```
main.py                      入口
train_cli.py                 命令行训练（AutoDL 云 GPU）
core/                        配置/日志/主题/缓存/依赖检测/异常
auth/                        认证与会话
data/                        数据集管理/格式转换/验证/增强
services/                    训练/评估/AI标注/导出/推理（QThread worker）
deployment/                  FastAPI 推理服务
gui/                         主窗口、工作流组件、标注编辑器、对话框
workspace/                   数据集/模型/输出/缓存（运行时生成，不入 git）
```
