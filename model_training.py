# model_training.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

class DNSModelTrainer:
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def load_engineered_data(self):
        """加载特征工程后的数据"""
        try:
            X_train = pd.read_pickle('data/processed/X_train.pkl')
            X_test = pd.read_pickle('data/processed/X_test.pkl')
            y_train = pd.read_pickle('data/processed/y_train.pkl')
            y_test = pd.read_pickle('data/processed/y_test.pkl')
            feature_columns = pd.read_pickle('data/processed/feature_columns.pkl')
            
            print(f"加载数据成功!")
            print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test, feature_columns
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None, None, None, None, None
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """训练多个模型进行比较"""
        print("\n训练机器学习模型...")
        
        # 定义模型
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # 训练并评估每个模型
        for name, model in models.items():
            print(f"\n训练 {name}...")
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 评估
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"{name} 准确率: {accuracy:.4f}")
            if auc:
                print(f"{name} AUC: {auc:.4f}")
            
            # 保存模型
            joblib.dump(model, f'models/{name.replace(" ", "_").lower()}_model.pkl')
        
        return self.results
    
    def evaluate_models(self, X_test, y_test):
        """全面评估模型性能"""
        print("\n" + "="*50)
        print("模型性能评估")
        print("="*50)
        
        # 比较所有模型
        comparison = []
        for name, result in self.results.items():
            comparison.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'AUC': result.get('auc', 'N/A')
            })
        
        comparison_df = pd.DataFrame(comparison)
        print("\n模型性能比较:")
        print(comparison_df)
        
        # 绘制性能比较图
        plt.figure(figsize=(10, 6))
        
        # 准确率比较
        plt.subplot(1, 2, 1)
        models = comparison_df['Model']
        accuracies = comparison_df['Accuracy']
        plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Model Accuracy Comparison')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # AUC比较（如果有）
        auc_values = [x if x != 'N/A' else 0 for x in comparison_df['AUC']]
        if any(auc != 0 for auc in auc_values):
            plt.subplot(1, 2, 2)
            plt.bar(models, auc_values, color=['skyblue', 'lightgreen', 'lightcoral'])
            plt.title('Model AUC Comparison')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 显示最佳模型的详细报告
        best_model_name = comparison_df.loc[comparison_df['Accuracy'].idxmax(), 'Model']
        print(f"\n最佳模型: {best_model_name}")
        
        best_result = self.results[best_model_name]
        print(f"\n{best_model_name} 详细分类报告:")
        print(classification_report(y_test, best_result['predictions']))
        
        # 绘制混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, best_result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Attacks'], 
                   yticklabels=['Benign', 'Attacks'])
        plt.title(f'{best_model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return comparison_df, best_model_name
    
    def run_model_training(self):
        """运行完整的模型训练流程"""
        print("DNS流量数据分析 - 阶段3: 模型训练")
        print("="*60)
        
        # 加载数据
        X_train, X_test, y_train, y_test, feature_columns = self.load_engineered_data()
        if X_train is None:
            return
        
        # 训练模型
        results = self.train_models(X_train, X_test, y_train, y_test)
        
        # 评估模型
        comparison_df, best_model = self.evaluate_models(X_test, y_test)
        
        print("\n" + "="*60)
        print("阶段3完成!")
        print(f"推荐使用模型: {best_model}")
        
        return {
            'comparison': comparison_df,
            'best_model': best_model,
            'feature_columns': feature_columns
        }

def main():
    trainer = DNSModelTrainer()
    results = trainer.run_model_training()
    
    if results:
        print(f"\n模型训练完成!")
        print(f"最佳模型: {results['best_model']}")
        print(f"使用的特征数量: {len(results['feature_columns'])}")

if __name__ == "__main__":
    main()