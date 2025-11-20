# create_directories.py
import os
from pathlib import Path

# 创建目录结构
directories = [
    "data/processed",
    "visualizations",
    "notebooks",
    "src",
    "output",
    "models"
]

for directory in directories:
    Path(directory).mkdir(parents=True, exist_ok=True)
    print(f"创建目录: {directory}")

print("目录结构创建完成!")