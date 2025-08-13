import os
import sys
import torch
import logging

# 添加src目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")

def check_model_files():
    """检查必要的模型文件"""
    required_files = [
        "model/pretrained/config.json",
        "model/pretrained/pytorch_model.bin",
        "model/pretrained/vocab.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

def main():
    # 检查模型文件
    missing_files = check_model_files()
    if missing_files:
        logger.error("模型文件缺失，请先运行 organize_model.py")
        logger.error("缺失文件:")
        for file in missing_files:
            logger.error(f" - {file}")
        
        # 尝试自动修复路径
        cache_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        if os.path.exists(cache_path):
            logger.info(f"检测到Hugging Face缓存目录: {cache_path}")
            logger.info("请运行: python organize_model.py")
        return
    
    # 启动GUI界面
    from gui_main import main as gui_main
    gui_main()

if __name__ == "__main__":
    main()