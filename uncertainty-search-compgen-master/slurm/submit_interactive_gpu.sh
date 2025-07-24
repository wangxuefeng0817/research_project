#!/bin/bash
# -*- coding: utf-8 -*-
#

# +
#!/bin/bash

# SBATCH --partition=gpu-h200-141g-short
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=140GB                      # 请求 200GB 内存
#SBATCH --time=16:00:00                  # 最大运行时间 24 小时
#SBATCH --cpus-per-gpu=8                # 每个 GPU 分配 8 个 CPU 核心
#SBATCH --output=logs/output_%j.log     # 保存标准输出日志
#SBATCH --error=logs/error_%j.log       # 保存标准错误日志

# 工作目录 
cd $WRKDIR

# 加载 Python 环境模块
module load scicomp-python-env

# 定义环境变量
BASEDIR=${BASEDIR:-$WRKDIR}
VENV=${VENV:-$WRKDIR/venv}              # 虚拟环境目录
PORT=${PORT:-8888}                      # Jupyter Notebook 的端口号

# 激活虚拟环境
if [ -f "${VENV}/bin/activate" ]; then
    source ${VENV}/bin/activate
else
    echo "虚拟环境未找到，请检查路径：${VENV}" >&2
    exit 1
fi

# 设置库路径（必要时）
export LD_LIBRARY_PATH=/appl/scibuilder-spack/aalto-rhel9-prod/2024-01/software/linux-rhel9-haswell/gcc-12.3.0/cuda-12.2.1-luiydfj/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${VENV}/lib/python3.11/site-packages/torch/lib/:$LD_LIBRARY_PATH

# 启动 Jupyter Notebook
cd $BASEDIR
jupyter notebook --port $PORT --NotebookApp.password='' --NotebookApp.token='' --ip=0.0.0.0 --no-browser

# -














