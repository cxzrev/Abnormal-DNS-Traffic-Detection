# feature_engineering.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DNSFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """加载阶段1处理后的数据"""
        try:
            combined_df = pd.read_pickle('data/processed/sample_data.pkl')
            print(f"成功加载数据: {combined_df.shape}")
            return combined_df
        except:
            print("无法找到处理后的数据文件，请先运行数据探索阶段")
            return None
    
    def remove_constant_features(self, df):
        """移除常量或几乎常量的特征"""
        print("\n移除常量特征...")
        
        # 计算每个特征的标准差
        numeric_features = df.select_dtypes(include=[np.number]).columns
        std_values = df[numeric_features].std()
        
        # 找出标准差为0的特征（常量特征）
        constant_features = std_values[std_values == 0].index.tolist()
        
        # 移除常量特征，但要保留label
        constant_features = [f for f in constant_features if f != 'label']
        
        if constant_features:
            print(f"移除常量特征: {constant_features}")
            df = df.drop(columns=constant_features)
        
        return df
    
    def process_categorical_features(self, df):
        """处理分类特征"""
        print("\n处理分类特征...")
        
        categorical_features = df.select_dtypes(include=['object']).columns
        
        for feature in categorical_features:
            print(f"处理分类特征: {feature}")
            
            # 对于集合类型的特征，提取集合大小作为新特征
            if df[feature].dtype == 'object' and df[feature].str.startswith('{').any():
                # 处理集合类型的特征，如 {'PTR'}
                df[f'{feature}_processed'] = df[feature].apply(
                    lambda x: len(eval(x)) if isinstance(x, str) and x != 'set()' else 0
                )
                
                # 对于rr_type，提取具体类型
                if feature == 'rr_type':
                    df['rr_type_main'] = df[feature].apply(
                        lambda x: list(eval(x))[0] if isinstance(x, str) and x not in ['set()', '{None}'] else 'None'
                    )
                    
                    # 对主要类型进行编码
                    le = LabelEncoder()
                    df['rr_type_encoded'] = le.fit_transform(df['rr_type_main'])
                    self.label_encoders['rr_type'] = le
                    
                    print(f"rr_type 分布: {df['rr_type_main'].value_counts()}")
            
            else:
                # 对于其他分类特征，直接进行标签编码
                le = LabelEncoder()
                df[f'{feature}_encoded'] = le.fit_transform(df[feature].fillna('Unknown'))
                self.label_encoders[feature] = le
        
        return df
    
    def create_new_features(self, df):
        """创建新的衍生特征"""
        print("\n创建衍生特征...")
        
        # 1. 创建频率特征的总和
        frequency_features = [col for col in df.columns if 'frequency' in col]
        if frequency_features:
            df['total_frequency'] = df[frequency_features].sum(axis=1)
        
        # 2. 创建TTL特征的比率
        if 'ttl_mean' in df.columns and 'ttl_variance' in df.columns:
            df['ttl_std'] = np.sqrt(df['ttl_variance'].clip(lower=0))
            df['ttl_coefficient_variation'] = df['ttl_std'] / (df['ttl_mean'] + 1e-6)  # 避免除0
        
        # 3. 创建域名复杂性特征
        if 'rr_name_length' in df.columns and 'rr_name_entropy' in df.columns:
            df['name_complexity'] = df['rr_name_length'] * df['rr_name_entropy']
        
        # 4. 创建记录类型多样性特征
        if 'rr_count' in df.columns and len(frequency_features) > 0:
            non_zero_freq = (df[frequency_features] > 0).sum(axis=1)
            df['record_type_diversity'] = non_zero_freq / (df['rr_count'] + 1e-6)
        
        print(f"创建了 {len([col for col in df.columns if col not in self.original_columns])} 个新特征")
        return df
    
    def prepare_model_data(self, df):
        """准备模型训练数据"""
        print("\n准备模型训练数据...")
        
        # 选择数值特征（排除标签和原始分类特征）
        numeric_features = df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_features 
                          if col != 'label' 
                          and not col.endswith('_original')]
        
        print(f"使用 {len(feature_columns)} 个特征进行建模")
        print("特征列表:", feature_columns)
        
        X = df[feature_columns]
        y = df['label']
        
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_columns, index=X.index)
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def analyze_feature_importance(self, df, feature_columns):
        """分析特征重要性（使用相关系数）"""
        print("\n分析特征与标签的相关性...")
        
        correlation_with_label = df[feature_columns + ['label']].corr()['label'].abs().sort_values(ascending=False)
        
        # 移除label自身的相关性
        correlation_with_label = correlation_with_label.drop('label')
        
        plt.figure(figsize=(10, 8))
        correlation_with_label.head(15).plot(kind='barh')
        plt.title('Correlation between Features and Attack Labels (abs)')
        plt.xlabel('Absolute Value of Correlation Coefficient')
        plt.tight_layout()
        plt.savefig('visualizations/feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("与攻击标签最相关的10个特征:")
        print(correlation_with_label.head(10))
        
        return correlation_with_label
    
    def run_feature_engineering(self):
        """运行完整的特征工程流程"""
        print("DNS流量数据分析 - 阶段2: 特征工程")
        print("="*60)
        
        # 加载数据
        df = self.load_data()
        if df is None:
            return
        
        self.original_columns = df.columns.tolist()
        
        # 步骤1: 移除常量特征
        df = self.remove_constant_features(df)
        
        # 步骤2: 处理分类特征
        df = self.process_categorical_features(df)
        
        # 步骤3: 创建新特征
        df = self.create_new_features(df)
        
        # 步骤4: 准备模型数据
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_model_data(df)
        
        # 步骤5: 分析特征重要性
        correlation_analysis = self.analyze_feature_importance(df, feature_columns)
        
        # 保存处理后的数据
        df.to_pickle('data/processed/engineered_data.pkl')
        X_train.to_pickle('data/processed/X_train.pkl')
        X_test.to_pickle('data/processed/X_test.pkl')
        y_train.to_pickle('data/processed/y_train.pkl')
        y_test.to_pickle('data/processed/y_test.pkl')
        
        # 保存特征列表
        pd.Series(feature_columns).to_pickle('data/processed/feature_columns.pkl')
        
        print("\n" + "="*60)
        print("阶段2完成!")
        print("下一步: 模型训练和评估")
        
        return {
            'X_train': X_train,
            'X_test': X_test, 
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': feature_columns,
            'correlation_analysis': correlation_analysis
        }

def main():
    engineer = DNSFeatureEngineer()
    results = engineer.run_feature_engineering()
    
    if results:
        print(f"\n数据准备完成，可以开始模型训练!")
        print(f"特征数量: {len(results['feature_columns'])}")
        print(f"训练样本: {len(results['X_train'])}")
        print(f"测试样本: {len(results['X_test'])}")

if __name__ == "__main__":
    main()