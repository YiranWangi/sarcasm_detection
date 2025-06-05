import argparse
from device import get_default_device
from bert_classifier import BertSarcasmClassifier
from t5_classifier import T5SarcasmClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, choices=["bert", "t5"], required=True,
        help="选择推理时使用哪个模型：bert / t5"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True,
        help="Bert 则是 .pth 文件；T5 则是保存模型的目录"
    )
    args = parser.parse_args()

    device = get_default_device()
    print("Using device:", device)

    if args.model_type == "bert":
        classifier = BertSarcasmClassifier(model_name="bert-base-chinese", device=device)
        classifier.load(args.checkpoint_path)
    else:  # t5
        classifier = T5SarcasmClassifier(model_name=args.checkpoint_path, device=device)

    samples = [
        "你好棒棒哦，椎间盘都没有你突出。",
        "今天天气真好。",
        "你这人真聪明，一点都不笨。",
    ]
    for sent in samples:
        result = classifier.predict(sent)
        if isinstance(result, tuple):
            # Bert 返回 (label, confidence)
            label, conf = result
            print(f"输入：{sent}\n → 预测：{label}，置信度：{conf:.2f}\n")
        else:
            # T5 返回 单一字符串
            print(f"输入：{sent}\n → 预测：{result}\n")

if __name__ == "__main__":
    main()