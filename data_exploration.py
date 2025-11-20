# data_exploration.py - 完整独立版本
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class DNSDataAnalyzer:
    def __init__(self, data_path="data/CSV"):
        self.data_path = Path(data_path)
        self.datasets = {}
        self.analysis_results = {}
        
    def extract_zip_files(self):
        """解压所有ZIP文件"""
        print("开始解压ZIP文件...")
        
        zip_files = list(self.data_path.rglob("*.zip"))
        print(f"找到 {len(zip_files)} 个ZIP文件")
        
        for zip_file in zip_files:
            extract_path = zip_file.parent / zip_file.stem
            if not extract_path.exists():
                print(f"解压: {zip_file}")
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    print(f"  解压到: {extract_path}")
                except Exception as e:
                    print(f"  解压失败: {e}")
            else:
                print(f"已存在: {extract_path}")
        return len(zip_files)
    
    def find_csv_files(self):
        """查找所有CSV文件"""
        print("\n查找CSV文件...")
        csv_files = list(self.data_path.rglob("*.csv"))
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        # 按类型分类
        attack_files = [f for f in csv_files if 'attack' in f.parent.name.lower()]
        benign_files = [f for f in csv_files if 'benign' in f.parent.name.lower() and 'attack' not in f.parent.name.lower()]
        
        print(f"攻击数据文件: {len(attack_files)} 个")
        print(f"正常数据文件: {len(benign_files)} 个")
        
        return attack_files, benign_files
    
    def load_sample_data(self, attack_files, benign_files, sample_size=5000):
        """加载样本数据进行初步分析"""
        print("\n加载样本数据...")
        
        # 加载攻击数据样本
        attack_samples = []
        for file in attack_files[:2]:  # 只加载前2个文件作为样本
            try:
                print(f"加载攻击文件: {file.name}")
                # 尝试不同的编码方式
                try:
                    df = pd.read_csv(file, nrows=sample_size)
                except UnicodeDecodeError:
                    df = pd.read_csv(file, nrows=sample_size, encoding='latin-1')
                
                df['label'] = 1  # 1表示攻击
                attack_samples.append(df)
                print(f"  成功加载 {len(df)} 行数据")
            except Exception as e:
                print(f"  加载失败: {e}")
                continue
        
        # 加载正常数据样本
        benign_samples = []
        for file in benign_files[:2]:  # 只加载前2个文件作为样本
            try:
                print(f"加载正常文件: {file.name}")
                # 尝试不同的编码方式
                try:
                    df = pd.read_csv(file, nrows=sample_size)
                except UnicodeDecodeError:
                    df = pd.read_csv(file, nrows=sample_size, encoding='latin-1')
                
                df['label'] = 0  # 0表示正常
                benign_samples.append(df)
                print(f"  成功加载 {len(df)} 行数据")
            except Exception as e:
                print(f"  加载失败: {e}")
                continue
        
        if attack_samples and benign_samples:
            attack_df = pd.concat(attack_samples, ignore_index=True)
            benign_df = pd.concat(benign_samples, ignore_index=True)
            combined_df = pd.concat([attack_df, benign_df], ignore_index=True)
            return combined_df, attack_df, benign_df
        else:
            print("错误: 无法加载足够的数据文件")
            return None, None, None
    
    def basic_data_analysis(self, df):
        """基础数据分析"""
        print("\n" + "="*50)
        print("基础数据分析")
        print("="*50)
        
        if df is None:
            print("没有数据可分析")
            return None
            
        # 基本信息
        print(f"数据集形状: {df.shape}")
        print(f"列数: {df.shape[1]}, 行数: {df.shape[0]}")
        
        # 列信息
        print(f"\n列名: {list(df.columns)}")
        print(f"\n数据类型:")
        print(df.dtypes.value_counts())
        
        # 缺失值分析
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        print(f"\n缺失值分析:")
        missing_columns = missing_data[missing_data > 0]
        if len(missing_columns) > 0:
            missing_info = pd.DataFrame({
                '缺失数量': missing_columns,
                '缺失比例%': missing_percent[missing_columns.index]
            })
            print(missing_info)
        else:
            print("没有缺失值")
        
        # 标签分布
        if 'label' in df.columns:
            print(f"\n标签分布:")
            label_counts = df['label'].value_counts()
            print(label_counts)
            if 1 in label_counts.index:
                attack_ratio = (label_counts[1] / len(df)) * 100
                print(f"攻击比例: {attack_ratio:.2f}%")
            else:
                print("没有攻击样本")
        
        return {
            'shape': df.shape,
            'columns': list(df.columns),
            'missing_columns': len(missing_columns),
            'label_distribution': label_counts if 'label' in df.columns else None
        }
    
    def statistical_analysis(self, df):
        """统计分析"""
        print("\n" + "="*50)
        print("统计分析")
        print("="*50)
        
        if df is None:
            print("没有数据可分析")
            return None
            
        # 数值列描述性统计
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print("数值列描述性统计:")
            print(df[numeric_cols].describe())
        else:
            print("没有数值列")
        
        # 分类列分析
        categorical_cols = df.select_dtypes(include=['object']).columns
        print(f"\n分类列: {list(categorical_cols)}")
        
        for col in categorical_cols[:3]:  # 只分析前3个分类列
            if col in df.columns:
                print(f"\n{col} 的唯一值数量: {df[col].nunique()}")
                if df[col].nunique() < 20:  # 如果唯一值较少，显示分布
                    print(f"{col} 的值分布:")
                    print(df[col].value_counts().head(10))
        
        return {
            'numeric_cols': list(numeric_cols),
            'categorical_cols': list(categorical_cols)
        }

def main():
    """主函数"""
    print("DNS流量数据分析 - 阶段1: 数据探索")
    print("="*60)
    
    # 初始化分析器
    analyzer = DNSDataAnalyzer("data/CSV")
    
    # 步骤1: 解压文件
    zip_count = analyzer.extract_zip_files()
    if zip_count == 0:
        print("没有找到ZIP文件，检查数据路径")
        return
    
    # 步骤2: 查找CSV文件
    attack_files, benign_files = analyzer.find_csv_files()
    
    if not attack_files and not benign_files:
        print("错误: 没有找到任何CSV文件!")
        return
    
    # 步骤3: 加载样本数据
    combined_df, attack_df, benign_df = analyzer.load_sample_data(attack_files, benign_files)
    
    if combined_df is None:
        print("错误: 无法加载数据!")
        return
    
    # 步骤4: 基础分析
    basic_info = analyzer.basic_data_analysis(combined_df)
    
    # 步骤5: 统计分析
    stats_info = analyzer.statistical_analysis(combined_df)
    
    # === 新增：保存处理后的数据 ===
    print("\n保存处理后的数据...")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存数据
    combined_df.to_pickle(processed_dir / "sample_data.pkl")
    attack_df.to_pickle(processed_dir / "attack_data.pkl") 
    benign_df.to_pickle(processed_dir / "benign_data.pkl")
    
    print(f"数据已保存到 {processed_dir}")
    # === 新增结束 ===
    
    print("\n" + "="*60)
    print("阶段1数据探索完成!")
    print("下一步: 基于探索结果进行特征工程")

if __name__ == "__main__":
    main()