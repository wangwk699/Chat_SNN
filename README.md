# Chat_SNN

## 安装环境

在另一台 Linux/NVIDIA GPU 服务器上，先克隆并进入本仓库，然后执行以下步骤。需要可用的 NVIDIA 驱动和 CUDA 12.4 支持的 GCC/G++ 编译器。

```bash
git clone git@github.com:wangwk699/Chat_SNN.git
cd Chat_SNN

conda create -n prefixquant python=3.9.21
conda activate prefixquant
conda install -c nvidia cuda-nvcc=12.4 cuda-cudart-dev=12.4 cuda-libraries-dev=12.4
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# --no-deps 不会补装构建依赖，先明确安装这些包。
python -m pip install packaging==25.0 ninja==1.11.1.4 setuptools==80.9.0 wheel==0.45.1

git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
# 固定为本次验证使用的上游版本。
git checkout e7706faf8d1c3b9f241e36860640ad1dac644ede
python -m pip install --use-pep517 --no-build-isolation --no-deps -e .
cd ..
pip install -r requirements.txt
```

关闭终端后重新使用环境时，需要先 `conda activate prefixquant`。上面的 CUDA 环境变量在当前 shell 中有效，重新编译扩展前应再次设置。`--use-pep517 --no-build-isolation` 避免旧 editable 安装流程再次进入找不到 PyTorch 的隔离环境。

安装后检查：

```bash
python -c 'import torch, fast_hadamard_transform; print(torch.__version__); print(torch.cuda.is_available()); print(fast_hadamard_transform.__file__)'
```

`fast-hadamard-transform` 是 editable 安装，安装后保留其源码目录。它作为外部依赖安装，不提交到本仓库。复现本次迁移验证时，可在其目录中先切换到上面记录的提交。

## 准备保存模型

本仓库只包含 **从保存的 T=16 Phase-SNN Qwen 模型进行多轮聊天**所需的代码和文档。

原工程当前脚本使用的模型位于其项目目录下：

```text
snn/T=16/Qwen-2.5-1.5B-Instruct/
```

请将这个目录的**全部文件**下载并上传至新服务器。建议保留模型子目录，例如 `/home/share/Qwen-2.5-1.5B-Instruct`；`/home/share` 只是示例，可以换成任意可读目录，通过运行命令的 `--model-path` 指定。

当前完整目录含：

```text
Qwen-2.5-1.5B-Instruct/
├── config.json
├── generation_config.json
├── prefixequant_config.json
├── prefixed_key_values.pth
├── pytorch_model.bin
├── tokenizer_config.json
├── special_tokens_map.json
├── added_tokens.json
├── vocab.json
└── merges.txt
```

权重约 3.55 GB（十进制），迁移整个目录即可；不需要下载原始 ANN 模型或额外的 `retrain.pth`。配置中的原服务器 `_name_or_path` 只是元数据，加载以 `--model-path` 指定的本地目录为准。

这是**已经做过离线旋转、带 Phase 神经元参数的 checkpoint**。从模型平台下载普通 Qwen 权重不能代替它。模型文件由你自行迁移，不放入 Git 仓库；`.gitignore` 已排除常见权重和缓存格式。

可以用仓库中记录的哈希检查上传完整性；将路径换成实际下载后的目录：

```bash
cd /home/share/Qwen-2.5-1.5B-Instruct
sha256sum -c /path/to/Chat_SNN/model_files.sha256
```

## 启动多轮聊天

先激活环境，再通过 Bash 传入模型目录。脚本默认使用当前环境的 `python`、GPU 0、T=16 和每卡 20GiB 的设备放置预算，使用当前激活的 Conda 环境。

```bash
conda activate prefixquant
bash /path/to/Chat_SNN/run_chat_snn_t16.sh \
  --model-path /home/share/Qwen-2.5-1.5B-Instruct
```

在仓库目录中可使用：

```bash
bash run_chat_snn_t16.sh --model-path /home/share/Qwen-2.5-1.5B-Instruct
```

第一个可选位置参数指定 GPU，后续参数直接转发到 Python：

```bash
# 使用 GPU 1，贪心生成，最多输出 128 个 token。
bash run_chat_snn_t16.sh 1 \
  --model-path /home/share/Qwen-2.5-1.5B-Instruct \
  --greedy --max-new-tokens 128

# 显式指定多张可见 GPU 和每卡设备放置预算。
bash run_chat_snn_t16.sh 0,1 \
  --model-path /home/share/Qwen-2.5-1.5B-Instruct \
  --max-memory 20GiB

# 不需要模型也可以查看参数说明。
bash run_chat_snn_t16.sh --help
```

`--model-path` 是必填参数，代码没有写死 `/home/share`。Bash 会自动进入仓库目录，可以从其他工作目录启动。需要显式选择解释器时，可设置 `PYTHON_BIN=/path/to/env/bin/python`。

默认使用模型 generation_config 的采样设置，每轮最多生成 256 个 token。可以通过 `--temperature`、`--top-p`、`--top-k`、`--system-prompt` 和 `--max-input-tokens` 调整。程序会为前缀 KV 和生成预留上下文空间，超长历史按 token 从左侧截断。

交互命令：

- 直接输入问题，回车生成；后续问题带上已有对话历史。
- `/clear`：清空历史，保留 system prompt。
- `exit` 或 `quit`：退出；Ctrl+C/EOF 也可退出。

## 运行实现与范围

加载流程为：读取本地 checkpoint → 创建 Qwen Phase 包装层 → 创建单粒度 FSNeuron 占位结构 → 再次读取 checkpoint 恢复 `d/tau/h/theta/v0` → 加载前缀 KV → 多轮生成。每轮复制基础前缀缓存，避免上轮生成污染前缀。

本项目针对当前 **Qwen-2.5-1.5B-Instruct、T=16、单粒度、浮点权重 Phase checkpoint** 提取。保留原路径的计算模块名和 checkpoint key。包装层外层 T=1，神经元内部 T=16。运行仍使用 GPU 上的 PyTorch 浮点计算和脉冲近似。

为减少不必要代码，未迁移训练、校准、数据集评测、粒度搜索、其他神经元、Llama、真实整数量化或旧聊天入口。Hadamard 辅助只保留该模型实际需要的 140 阶矩阵及二次幂路径；任意其他模型即使配置 T=16，也不能视为已兼容。加载器会检查架构、T 和神经元参数是否存在，避免误用普通 ANN checkpoint。

模型目录变更只需修改启动命令。更换模型架构、时间步或多粒度 checkpoint 需要相应适配，不能只改目录名。模型由本地文件加载，不依赖原服务器的数据集、cache 或日志目录，也不会重复执行离线旋转。

目录说明见 [代码结构总结.md](代码结构总结.md)。源码来源和保留的许可证见 [NOTICE](NOTICE) 与 [LICENSE](LICENSE)。

## 验证记录

迁移验证使用原服务器上的上述 1.5B 保存模型、Python 3.9、PyTorch 2.4.1+cu124、Transformers 4.40.1、Accelerate 0.28.0 和 NVIDIA A100。迁移前后对照中，1265 个神经元参数张量完全一致，两轮贪心生成的输出 token 和完整 logits 逐元素一致（最大绝对差为 0）；Python 编译、命令行导入和 Bash 语法检查通过；从仓库外通过 Bash 启动、含空格模型路径、连续对话、`/clear` 和退出也已实测通过。新服务器安装环境仍需自行执行导入和聊天检查。
