# Sarcasm Detection Project

本项目演示了多种基于 Transformer 的中文“反讽检测”思路，包括：
- 最简单的 `BertForSequenceClassification` 二分类；
- Seq2Seq（T5/MT5）生成式分类；
- 在 Transformer 输出之上叠加 CNN / CRF 的高级模型。

每个模块都已拆分到不同文件，目录结构如下：
sarcasm_detection/modifiedmodel/
├── device.py # 设备选择：CPU / MPS（Apple Silicon）
├── bert_classifier.py # 基于 Hugging Face BertForSequenceClassification 的二分类器
├── t5_classifier.py # 基于 T5ForConditionalGeneration 的生成式分类器
├── advanced_models.py # Transformer + CNN / CRF 的自定义模型定义
├── train_bert.py # Bert 分类模型的训练脚本
├── train_t5.py # T5 生成式模型的训练脚本
└── predict.py # 各种模型的推理示例