import json
import os

def load_dataset(data_type="all"):
    """加载指定类型的素材数据"""
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    datasets = {}
    
    if data_type in ["quotes", "all"]:
        with open(os.path.join(data_dir, 'quotes.json'), 'r', encoding='utf-8') as f:
            datasets['quotes'] = json.load(f)
    
    if data_type in ["examples", "all"]:
        with open(os.path.join(data_dir, 'examples.json'), 'r', encoding='utf-8') as f:
            datasets['examples'] = json.load(f)
    
    if data_type in ["poems", "all"]:
        with open(os.path.join(data_dir, 'poems.json'), 'r', encoding='utf-8') as f:
            datasets['poems'] = json.load(f)
    
    return datasets