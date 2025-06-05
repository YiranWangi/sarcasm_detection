# Sarcasm Detection Project

本项目主要基于 BERT 和 T5 模型，实现中文（MT5）与英文（BERT）两种不同思路的讽刺（Sarcasm）检测（分类）任务。项目包括数据预处理、模型训练、评估以及推理代码，能够方便地复现训练流程并进行在线/离线推断。

## 项目结构

每个模块都已拆分到不同文件，目录结构如下：
sarcasm_detection/ 
├── README.md # 项目说明文件（当前文件）
├── device.py # 设备选择工具（支持 MPS / CPU）
├── bert_classifier.py # 使用 Hugging Face Transformers 的 BERT 分类器封装
├── t5_classifier.py # 基于 T5/MT5 的分类器封装，适用于文本生成式分类任务 
├── advanced_models.py # 自定义的高级模型定义（BERT+CRF、BERT+CNN 等）
├── train_bert.py # 训练 BERT 分类器的脚本（包含数据加载、训练、验证到保存）
├── train_t5.py # 训练 T5/MT5 分类器的脚本（包含数据加载、训练、验证到保存）
└── predict.py # 各种模型的推理示例 

## 环境依赖

执行以下命令即可安装本项目所需的最小依赖（基于 `requirements.txt`）：
```bash
pip install -r requirements.txt
```

## 数据准备
1. 确保项目根目录下存在 scrcasm_data.xlsx，表格需包含两列：  
text：待分类的句子文本  
label：整数标签（0 表示“非反讽”，1 表示“反讽”）
2. train_bert.py 与 train_t5.py 会自动读取该文件并进行数据切分（80% 训练 / 10% 验证 / 10% 测试，按照标签比例分层抽样）

## 输出示例
1. T5模型
```bash
Reloaded modules: device, t5_classifier
Train/Val/Test = 3200/400/400
Using device: mps
/opt/anaconda3/lib/python3.12/site-packages/transformers/tokenization_utils_base.py:3959: UserWarning: `as_target_tokenizer` is deprecated and will be removed in v5 of Transformers. You can tokenize your labels by using the argument `text_target` of the regular `__call__` method (either in the same call as your input texts if you use the same keyword arguments, or in a separate call.
  warnings.warn(
Epoch 1/10  Train Loss: 2.5135
Validation Accuracy: 1.0000
Epoch 2/10  Train Loss: 0.4929
Validation Accuracy: 1.0000
Epoch 3/10  Train Loss: 0.1916
Validation Accuracy: 1.0000
Epoch 4/10  Train Loss: 0.0209
Validation Accuracy: 1.0000
Epoch 5/10  Train Loss: 0.0089
Validation Accuracy: 1.0000
Epoch 6/10  Train Loss: 0.0075
Validation Accuracy: 1.0000
Epoch 7/10  Train Loss: 0.0034
Validation Accuracy: 1.0000
Epoch 8/10  Train Loss: 0.0013
Validation Accuracy: 1.0000
```

2. Bert模型
```bash
训练集样本数：3200，验证集样本数：400，测试集样本数：400
Using device: mps
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-chinese and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Epoch 1/3  Train Loss: 0.0574
Validation Accuracy: 1.0000
Epoch 2/3  Train Loss: 0.0006
Validation Accuracy: 1.0000
Epoch 3/3  Train Loss: 0.0002
Validation Accuracy: 1.0000
Saved best_bert.pth
```

## 联系方式
如有问题或建议，可联系项目作者：
邮箱：yiran.wang@yale.edu
GitHub：@YiranWangi







