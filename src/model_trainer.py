import os
import json
import torch
import logging,shutil
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import losses, InputExample
from .data_processor import DataProcessor
from .model_loader import ModelLoader
from tqdm import tqdm
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelTrainer")

class MaterialDataset(Dataset):
    """自定义素材数据集，适配Sentence-BERT训练格式"""
    def __init__(self, datasets):
        self.samples = []
        
        # 为每个素材创建InputExample对象
        for data_type, items in datasets.items():
            for item in items:
                # 创建模型输入文本
                text = f"{item['content']} [SEP] {' '.join(item['keywords'])}"
                if 'theme' in item:
                    text += f" [SEP] {item['theme']}"
                
                # 使用清理后的文本作为锚点文本
                # 创建两个相同的文本作为输入
                self.samples.append(InputExample(
                    texts=[item['cleaned_text'], item['cleaned_text']],  # 两个相同的文本
                    label=1.0  # 固定标签值
                ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

def train_model(epochs=3, batch_size=16, use_cuda=True, iteration=1, total_iterations=3):
    """训练模型的主函数 - 支持多次迭代训练"""
    # 确定设备
    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 加载和处理数据
    processor = DataProcessor()
    datasets = processor.load_and_preprocess()
    
    # 创建数据集
    dataset = MaterialDataset(datasets)
    logger.info(f"训练数据集大小: {len(dataset)} 个样本")
    
    # 加载模型 - 如果是后续迭代，加载前一次训练的模型
    model_loader = ModelLoader()
    
    if iteration > 1 and os.path.exists("model/fine_tuned"):
        logger.info(f"加载前一次迭代的模型 (迭代 #{iteration-1})")
        model, device = model_loader.load_model(use_fine_tuned=True, device=device)
    else:
        logger.info("加载预训练模型")
        model, device = model_loader.load_model(use_fine_tuned=False, device=device)
    
    # 训练配置
    train_dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    
    # 使用更适合无监督/自监督任务的损失函数
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # 微调模型
    logger.info(f"开始微调模型 (迭代 #{iteration}/{total_iterations})")
    
    # 禁用AMP - 避免精度问题
    use_amp = False
    
    try:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            steps_per_epoch=len(train_dataloader),
            warmup_steps=min(100, len(train_dataloader)),
            output_path=f"model/temp_iter{iteration}",
            show_progress_bar=True,
            checkpoint_path=f"model/checkpoints_iter{iteration}",
            checkpoint_save_steps=min(100, len(train_dataloader)),
            checkpoint_save_total_limit=3,
            use_amp=use_amp,
            optimizer_params={'lr': 2e-5 * (0.8 ** (iteration-1))},  # 逐步降低学习率
            scheduler='warmupcosine'
        )
    except TypeError as e:
        # 兼容模式
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=min(100, len(train_dataloader)),
            output_path=f"model/temp_iter{iteration}",
            show_progress_bar=True,
            checkpoint_path=f"model/checkpoints_iter{iteration}",
            checkpoint_save_steps=min(100, len(train_dataloader)),
            checkpoint_save_total_limit=3,
            use_amp=use_amp,
            optimizer_params={'lr': 2e-5 * (0.8 ** (iteration-1))},
            scheduler='warmupcosine'
        )
    
    # 保存微调后的模型
    model_loader.save_model(model)
    
    # 如果是最后一次迭代，生成并保存嵌入向量
    if iteration == total_iterations:
        logger.info("最后一次迭代，生成素材嵌入向量...")
        
        # 收集所有清理后的文本
        all_texts = []
        metadata = []
        for data_type, items in datasets.items():
            for item in items:
                all_texts.append(item['cleaned_text'])
                metadata.append({
                    'type': data_type,
                    'content': item['content'],
                    'source': item.get('source', ''),
                    'keywords': item['keywords'],
                    'theme': item.get('theme', '')
                })
        
        # 分批编码以减少内存使用
        embeddings = []
        batch_size_emb = 128
        
        # 确保模型在正确设备上
        model.to(device)
        
        # 禁用梯度计算以节省内存
        with torch.no_grad():
            for i in tqdm(range(0, len(all_texts), batch_size_emb), desc="生成嵌入"):
                batch = all_texts[i:i+batch_size_emb]
                batch_emb = model.encode(
                    batch, 
                    convert_to_tensor=True, 
                    device=device,
                    show_progress_bar=False,
                    batch_size=batch_size_emb
                )
                embeddings.append(batch_emb.cpu())  # 移到CPU以节省GPU内存
        
        # 合并所有嵌入
        embeddings = torch.cat(embeddings, dim=0)
        torch.save(embeddings, "model/embeddings.pt")
        
        # 保存元数据
        with open("model/metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练完成! 保存嵌入向量: {embeddings.shape[0]} 条, 维度: {embeddings.shape[1]}")
    
    logger.info(f"迭代 #{iteration} 完成! 模型已保存到 model/fine_tuned")
    return model

def iterative_training(total_iterations=3, epochs_per_iter=3, batch_size=16):
    """执行多次迭代训练"""
    best_model = None
    
    for i in range(1, total_iterations+1):
        logger.info(f"\n{'='*40}")
        logger.info(f"开始训练迭代 #{i}/{total_iterations}")
        logger.info(f"{'='*40}")
        
        model = train_model(
            epochs=epochs_per_iter,
            batch_size=batch_size,
            use_cuda=True,
            iteration=i,
            total_iterations=total_iterations
        )
        
        # 保存当前迭代的模型副本
        shutil.copytree(
            "model/fine_tuned", 
            f"model/fine_tuned_iter{i}",
            dirs_exist_ok=True
        )
        
        # 评估当前模型（可选）
        # accuracy = evaluate_model(model)
        # if best_model is None or accuracy > best_accuracy:
        #     best_model = model
        #     best_accuracy = accuracy
    
    logger.info(f"所有 {total_iterations} 次迭代训练完成!")
    # 使用最后一次训练的模型作为最终模型
    return model

if __name__ == "__main__":
    try:
        # 设置环境变量
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # 执行多次迭代训练
        iterative_training(
            total_iterations=5,  # 训练10次
            epochs_per_iter=30,   # 每次20个epoch
            batch_size=32        # 批处理大小32
        )
    except Exception as e:
        logger.exception(f"训练过程中发生严重错误: {str(e)}")
        # 提供更友好的错误信息
        if "CUDA out of memory" in str(e):
            logger.error("显存不足! 请尝试减小批处理大小:")
            logger.error("1. 修改 train_model 调用中的 batch_size 参数")
            logger.error("2. 例如: train_model(epochs=3, batch_size=8, use_cuda=True)")
        elif "too many dimensions" in str(e):
            logger.error("数据格式错误! 请检查数据集格式是否正确")
            logger.error("确保每个数据项包含 'content' 和 'keywords' 字段")
        else:
            logger.error("未知错误，请查看详细日志")
        raise