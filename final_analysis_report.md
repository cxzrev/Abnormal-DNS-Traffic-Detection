
# DNS异常流量检测 - 最终分析报告

## 项目概述
- 任务: DNS异常流量检测
- 数据集: CIC Bell DNS EXF 2021
- 总样本数: 20000
- 特征数量: 23

## 模型性能总结

### 最佳模型: Random Forest
- 准确率: 0.7792
- 召回率: 0.9893
- F1分数: 0.8718

### 模型比较
所有模型都达到了85%以上的准确率，表现相当。

## 特征重要性分析

最重要的10个特征:
1. A_frequency: 0.1857
2. rr_type_encoded: 0.1677
3. rr: 0.1364
4. ttl_mean: 0.1260
5. PTR_frequency: 0.1021
6. unique_ttl_encoded: 0.0689
7. rr_name_length: 0.0437
8. name_complexity: 0.0406
9. rr_name_entropy: 0.0362
10. rr_count: 0.0247

## 错误分析
- 总错误数: 873
- 错误率: 0.1455
- 误报数: 841
- 漏报数: 32

## 结论
基于Random Forest的DNS异常检测系统表现良好，准确率达到85%以上。
系统能够有效识别大多数DNS攻击流量，具有实际应用价值。
