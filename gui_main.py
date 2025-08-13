import sys
import os,torch
import logging
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QComboBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QStatusBar,
    QTextEdit, QSplitter, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QTimer
from src.gui_interface import GUIInterface

class MaterialSearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("作文素材智能检索系统")
        self.setGeometry(100, 100, 1440, 720)
        
        # 初始化界面
        self.init_ui()
        
        # 初始化搜索接口
        self.search_interface = GUIInterface(model_dir="model")
        self.load_model()
        
        # 配置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("GUI")
        self.logger.addHandler(self.log_handler)
        
        # 初始化设备信息
        self.device_type = "CPU"  # 默认为CPU
        if torch.cuda.is_available():
            self.device_type = f"GPU ({torch.cuda.get_device_name(0)})"
    
    def init_ui(self):
        """初始化用户界面"""
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # 搜索区域
        search_layout = QHBoxLayout()
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入关键词或描述...")
        self.search_input.setMinimumHeight(40)
        self.search_input.returnPressed.connect(self.do_search)
        
        self.category_combo = QComboBox()
        self.category_combo.addItems(["全部类型", "名言", "事例", "古诗文"])
        
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 100)
        self.count_spin.setValue(5)
        self.count_spin.setToolTip("结果数量")
        
        self.search_btn = QPushButton("搜索")
        self.search_btn.setMinimumHeight(40)
        self.search_btn.clicked.connect(self.do_search)
        
        search_layout.addWidget(self.search_input, 5)
        search_layout.addWidget(QLabel("类型:"))
        search_layout.addWidget(self.category_combo, 1)
        search_layout.addWidget(QLabel("数量:"))
        search_layout.addWidget(self.count_spin, 1)
        search_layout.addWidget(self.search_btn, 1)
        
        # 结果表格
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)  # 增加序号列
        self.results_table.setHorizontalHeaderLabels(["序号", "类型", "内容", "来源", "标签", "相关度"])
        self.results_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # 内容列自适应
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.results_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.results_table.doubleClicked.connect(self.show_result_details)
        
        # 设置列宽策略
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)  # 序号列
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)  # 类型列
        self.results_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)  # 来源列
        self.results_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)  # 标签列
        self.results_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)  # 相关度列
        
        # 日志区域
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(150)
        self.log_text.setStyleSheet("background-color: #f0f0f0;")
        
        # 创建日志处理器
        class TextEditLogger(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                self.text_widget.append(msg)
                self.text_widget.verticalScrollBar().setValue(
                    self.text_widget.verticalScrollBar().maximum()
                )
        
        self.log_handler = TextEditLogger(self.log_text)
        self.log_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # 创建分割器
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.results_table)
        splitter.addWidget(self.log_text)
        splitter.setSizes([500, 200])
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("正在初始化...")
        
        # 添加设备标签到状态栏
        self.device_label = QLabel("设备: 检测中...")
        self.status_bar.addPermanentWidget(self.device_label)
        
        # 添加进度条到状态栏（默认隐藏）
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # 无限进度条
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # 组装主布局
        main_layout.addLayout(search_layout)
        main_layout.addWidget(splitter)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
    
    def load_model(self):
        """加载模型"""
        self.status_bar.showMessage("正在加载模型，请稍候...")
        self.search_btn.setEnabled(False)
        
        # 更新设备信息
        self.device_label.setText(f"设备: {self.device_label}")
        
        self.loader_thread = self.search_interface.load_model_async()
        self.loader_thread.loaded.connect(self.on_model_loaded)
        self.loader_thread.error.connect(self.on_model_error)
        self.loader_thread.start()
    
    def on_model_loaded(self, engine, has_fine_tuned, has_embeddings):
        """模型加载完成"""
        self.search_interface.engine = engine
        self.search_btn.setEnabled(True)
        
        # 更新状态信息
        model_type = "微调模型" if has_fine_tuned else "预训练模型"
        embeddings = "预计算嵌入" if has_embeddings else "实时编码"
        
        status_msg = f"就绪 | 模型: {model_type} | 编码: {embeddings}"
        self.status_bar.showMessage(status_msg)
        
        # 更新设备标签
        self.device_label.setText(f"设备: {self.device_type}")
        
        self.logger.info(f"模型加载完成 | 设备: {self.device_type} | 模型类型: {model_type} | 编码方式: {embeddings}")
    
    def on_model_error(self, error_msg):
        """模型加载错误"""
        self.status_bar.showMessage(f"加载失败: {error_msg}")
        self.logger.error(f"模型加载错误: {error_msg}")
    
    def do_search(self):
        """执行搜索操作"""
        query = self.search_input.text().strip()
        if not query:
            return
            
        # 获取搜索参数
        category_map = {
            "全部类型": "all",
            "名言": "quotes",
            "事例": "examples",
            "古诗文": "poems"
        }
        category = category_map.get(self.category_combo.currentText(), "all")
        top_k = self.count_spin.value()
        
        # 记录开始时间
        start_time = time.time()
        
        # 如果是CPU设备，显示等待提示
        if "GPU" not in self.device_type:
            self.status_bar.showMessage("正在搜索，请稍候...")
            self.progress_bar.setVisible(True)
            self.search_btn.setEnabled(False)
            QApplication.processEvents()  # 更新UI
        
        self.logger.info(f"搜索: '{query}' 类型: {category} 数量: {top_k}")
        
        # 执行搜索
        results = self.search_interface.search(
            query, 
            category=category, 
            top_k=top_k
        )
        
        # 计算处理时间
        elapsed_time = time.time() - start_time
        time_msg = f"搜索完成 | 耗时: {elapsed_time:.2f}秒 | 设备: {self.device_type}"
        
        # 隐藏进度条（如果是CPU）
        if "GPU" not in self.device_type:
            self.progress_bar.setVisible(False)
            self.search_btn.setEnabled(True)
        
        # 显示结果
        self.show_results(results)
        
        if results:
            self.status_bar.showMessage(f"找到 {len(results)} 条结果 | {time_msg}")
            self.logger.info(f"找到 {len(results)} 条相关素材 | {time_msg}")
        else:
            self.status_bar.showMessage(f"未找到相关素材 | {time_msg}")
            self.logger.info(f"未找到相关素材 | {time_msg}")
    
    def show_results(self, results):
        """在表格中显示搜索结果"""
        self.results_table.setRowCount(len(results) if results else 1)
        
        if not results:
            # 显示无结果提示
            self.results_table.setItem(0, 0, QTableWidgetItem(""))
            self.results_table.setItem(0, 1, QTableWidgetItem("无结果"))
            self.results_table.setItem(0, 2, QTableWidgetItem("未找到匹配的素材，请尝试其他关键词"))
            self.results_table.setItem(0, 3, QTableWidgetItem(""))
            self.results_table.setItem(0, 4, QTableWidgetItem(""))
            self.results_table.setItem(0, 5, QTableWidgetItem(""))
            
            # 合并单元格
            self.results_table.setSpan(0, 1, 1, 5)
            return
        
        # 清除可能的合并单元格
        self.results_table.clearSpans()
        
        for i, res in enumerate(results):
            # 序号
            self.results_table.setItem(i, 0, QTableWidgetItem(str(i+1)))
            self.results_table.item(i, 0).setTextAlignment(Qt.AlignCenter)
            
            # 类型
            self.results_table.setItem(i, 1, QTableWidgetItem(res['type']))
            
            # 内容 (限制长度)
            content = res['content']
            display_content = content
            if len(content) > 100:  # 限制为100个字符
                display_content = content[:97] + "..."
            self.results_table.setItem(i, 2, QTableWidgetItem(display_content))
            self.results_table.item(i, 2).setToolTip(content)  # 悬停提示显示完整内容
            self.results_table.item(i, 2).setData(Qt.UserRole, content)  # 存储完整内容
            
            # 来源
            source = res.get('source', '无')
            self.results_table.setItem(i, 3, QTableWidgetItem(source))
            
            # 标签
            tags = ", ".join(res['tags'])
            display_tags = tags
            if len(tags) > 50:  # 限制标签显示长度
                display_tags = tags[:47] + "..."
            self.results_table.setItem(i, 4, QTableWidgetItem(display_tags))
            self.results_table.item(i, 4).setToolTip(tags)  # 悬停提示显示完整标签
            
            # 相关度
            score_item = QTableWidgetItem(f"{res['score']:.4f}")
            score_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.results_table.setItem(i, 5, score_item)
        
        # 调整列宽
        self.results_table.resizeColumnsToContents()
    
    def show_result_details(self, index):
        """显示结果的完整详情"""
        # 检查是否是无结果行
        if self.results_table.rowCount() == 1 and self.results_table.item(0, 1).text() == "无结果":
            return
        
        row = index.row()
        content = self.results_table.item(row, 2).data(Qt.UserRole)
        material_type = self.results_table.item(row, 1).text()
        source = self.results_table.item(row, 3).text()
        tags = self.results_table.item(row, 4).toolTip()  # 获取完整标签
        score = self.results_table.item(row, 5).text()
        
        # 创建详情对话框
        detail_dialog = QMessageBox(self)
        detail_dialog.setWindowTitle("素材详情")
        detail_dialog.setIcon(QMessageBox.Information)
        detail_dialog.setMinimumWidth(600)  # 增加对话框宽度
        
        # 格式化详情内容
        detail_text = f"<b>序号:</b> {row+1}<br>"
        detail_text += f"<b>类型:</b> {material_type}<br>"
        detail_text += f"<b>相关度:</b> {score}<br>"
        detail_text += f"<b>来源:</b> {source}<br>"
        detail_text += f"<b>标签:</b> {tags}<br><br>"
        detail_text += f"<b>内容:</b><br>{content}"
        
        detail_dialog.setTextFormat(Qt.RichText)
        detail_dialog.setText(detail_text)
        
        # 添加复制按钮
        copy_btn = detail_dialog.addButton("复制内容", QMessageBox.ActionRole)
        close_btn = detail_dialog.addButton(QMessageBox.Close)
        
        # 设置默认按钮为关闭
        detail_dialog.setDefaultButton(close_btn)
        
        # 显示对话框
        detail_dialog.exec_()
        
        # 处理复制操作
        if detail_dialog.clickedButton() == copy_btn:
            clipboard = QApplication.clipboard()
            clipboard.setText(content)
            self.logger.info(f"素材内容已复制到剪贴板 (序号: {row+1})")

def main():
    app = QApplication(sys.argv)
    window = MaterialSearchApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()