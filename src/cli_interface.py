import torch,os
class CLIInterface:
    def __init__(self, model_dir="model", use_fine_tuned=True):
        from .semantic_search import SemanticSearchEngine
        from .model_loader import ModelLoader
        
        # 自动选择设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {'GPU加速' if device == 'cuda' else 'CPU运行'}")
        
        self.engine = SemanticSearchEngine(
            model_dir=model_dir,
            use_fine_tuned=use_fine_tuned,
            device=device
        )
        
        # 检查是否有微调模型可用
        model_loader = ModelLoader(model_dir)
        self.has_fine_tuned = os.path.exists(os.path.join(model_dir, "fine_tuned"))
        self.has_embeddings = os.path.exists(os.path.join(model_dir, "embeddings.pt"))
        
        if self.has_fine_tuned and self.has_embeddings:
            print("使用微调模型和预计算嵌入向量")
        elif self.has_fine_tuned:
            print("使用微调模型（实时编码）")
        else:
            print("使用预训练模型（实时编码）")
    
    def run(self):
        print("\n=== 作文素材智能检索工具 ===")
        print("支持检索类型: 名言(1) 事例(2) 古诗文(3) 全部(4)")
        
        while True:
            try:
                query = input("\n请输入关键词或描述 (输入q退出): ").strip()
                if query.lower() in ['q', 'quit', 'exit']:
                    break
                if not query:
                    continue
                
                cat_input = input("请选择类型 (1-4, 默认4): ").strip() or "4"
                category_map = {"1": "quotes", "2": "examples", "3": "poems", "4": "all"}
                category = category_map.get(cat_input, "all")
                
                # 添加相似度阈值设置
                #threshold_input = input("相似度阈值 (0.0-1.0, 默认0.3): ").strip()
                #similarity_threshold = float(threshold_input) if threshold_input else 0.3
                similarity_threshold = 0.0

                #结果数
                count_input = input("结果数 (默认5): ").strip() or "5"
                try:
                    top_k = int(count_input)
                except ValueError:
                    print("无效的结果数，使用默认值5")
                    top_k = 5

                results = self.engine.search(
                    query, 
                    top_k=top_k, 
                    category=category,
                    similarity_threshold=similarity_threshold
                )
                
                if not results:
                    print("\n未找到相关素材")
                    continue
                
                print(f"\n找到 {len(results)} 条相关素材:")
                for i, res in enumerate(results, 1):
                    print(f"\n[{i}] {res['type']}: {res['content']}")
                    print(f"   来源: {res.get('source', '无')}")
                    print(f"   标签: {', '.join(res['tags'])}")
                    print(f"   相关度: {res['score']:.4f}")
            
            except KeyboardInterrupt:
                print("\n操作已取消")
                break
            except Exception as e:
                print(f"发生错误: {str(e)}")