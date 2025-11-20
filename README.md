# Abnormal-DNS-Traffic-Detection

## 项目概述

这是一个基于机器学习的DNS异常流量检测系统，使用CIC Bell DNS EXF 2021数据集进行训练和评估。项目实现了从数据探索、特征工程到模型训练和结果分析的完整流程。

## 技术栈

### 编程语言
- **Python** 3.12.6

### 主要依赖包
- `pandas>=2.2.3` - 数据处理和分析
- `numpy>=2.2.5` - 数值计算
- `matplotlib>=3.10.3` - 数据可视化
- `seaborn>=0.13.2` - 统计可视化
- `scikit-learn>=1.7.2` - 机器学习算法
- `jupyter>=1.1.1` - 交互式笔记本
- `notebook>=7.5.0` - Jupyter笔记本环境

## 数据集

### 数据来源
本项目使用 **CIC Bell DNS EXF 2021** 数据集，该数据集由加拿大网络安全研究所（Canadian Institute for Cybersecurity）提供。

### 数据引用
根据数据集许可要求，任何使用或重新分发数据必须包含对CIC-Bell-DNS-EXF-2021数据集和以下论文的引用：

> Samaneh Mahdavifar, Amgad Hanafy Salem, Princy Victor, Miguel Garzon, Amir H. Razavi, Natasha Hellberg, Arash Habibi Lashkari, "Lightweight Hybrid Detection of Data Exfiltration using DNS based on Machine Learning", The 11th IEEE International Conference on Communication and Network Security (ICCNS), Dec. 3-5, 2021, Beijing Jiaotong University, Weihai, China.


### 数据统计
- 总样本数: 20,000
- 攻击样本: 10,000 (50%)
- 正常样本: 10,000 (50%)
- 原始特征数: 28个
- 处理后特征数: 23个

## 安装和运行

### 1. 环境配置

```bash
# 克隆项目
git clone <repository-url>
cd DNS-Anomaly-Detection

# 安装依赖
pip install -r requirements.txt

# 阶段1: 数据探索
python data_exploration.py

# 阶段2: 特征工程
python feature_engineering.py

# 阶段3: 模型训练
python model_training.py

# 阶段4: 结果分析
python result_analysis.py
