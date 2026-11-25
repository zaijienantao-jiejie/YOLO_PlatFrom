# -*- coding: utf-8 -*-
"""系统诊断对话框：环境检查、硬件检测、依赖完整性"""
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                               QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
                               QFormLayout, QTextEdit)

from core.dependencies import check_dependencies, torch_device_info, system_info, is_full_install


class DiagnosticDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("系统诊断")
        self.resize(640, 700)
        lay = QVBoxLayout(self)

        bar = QHBoxLayout()
        title = QLabel("🔧 系统诊断")
        title.setObjectName("title")
        bar.addWidget(title)
        bar.addStretch(1)
        btn_refresh = QPushButton("🔄 重新检测")
        btn_refresh.clicked.connect(self.run_diagnosis)
        bar.addWidget(btn_refresh)
        lay.addLayout(bar)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        lay.addWidget(self.summary_label)

        grp = QGroupBox("环境信息")
        self.env_form = QFormLayout(grp)
        lay.addWidget(grp)

        grp2 = QGroupBox("依赖完整性")
        glay = QVBoxLayout(grp2)
        self.deps_table = QTableWidget(0, 2)
        self.deps_table.setHorizontalHeaderLabels(["依赖", "版本"])
        self.deps_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        glay.addWidget(self.deps_table)
        lay.addWidget(grp2, 1)

        grp3 = QGroupBox("硬件信息")
        self.hw_text = QTextEdit()
        self.hw_text.setReadOnly(True)
        hlay = QVBoxLayout(grp3)
        hlay.addWidget(self.hw_text)
        lay.addWidget(grp3)

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.accept)
        lay.addWidget(btn_close)

        self.run_diagnosis()

    def run_diagnosis(self):
        deps = check_dependencies()
        torch_info = torch_device_info()
        sys_info = system_info()

        # 依赖表
        self.deps_table.setRowCount(0)
        missing = []
        for name, ver in deps.items():
            row = self.deps_table.rowCount()
            self.deps_table.insertRow(row)
            self.deps_table.setItem(row, 0, QTableWidgetItem(name))
            if ver:
                self.deps_table.setItem(row, 1, QTableWidgetItem(f"✓ {ver}"))
            else:
                self.deps_table.setItem(row, 1, QTableWidgetItem("✗ 未安装"))
                missing.append(name)

        # 环境信息
        while self.env_form.rowCount():
            self.env_form.removeRow(0)
        self.env_form.addRow("操作系统", QLabel(sys_info["os"]))
        self.env_form.addRow("架构", QLabel(sys_info["machine"]))
        if torch_info["available"]:
            self.env_form.addRow("GPU", QLabel(f"✓ {torch_info['gpu']} (CUDA {torch_info['cuda_version']})"))
        else:
            self.env_form.addRow("GPU", QLabel("✗ 无可用 CUDA GPU（将使用 CPU 训练）"))
        self.env_form.addRow("CPU", QLabel(f"{sys_info['cpu']} ({sys_info['cpu_cores']} 线程)"))
        self.env_form.addRow("内存", QLabel(f"{sys_info['ram_gb']} GB" if sys_info["ram_gb"] else "未知"))
        self.env_form.addRow("磁盘剩余", QLabel(f"{sys_info['disk_free_gb']} GB" if sys_info["disk_free_gb"] else "未知"))

        # 硬件文本
        self.hw_text.setPlainText(
            f"设备: {torch_info['device']}\n"
            f"GPU: {torch_info['gpu'] or '无'}\n"
            f"CUDA: {torch_info['cuda_version'] or '无'}\n"
            f"CPU: {sys_info['cpu']} ({sys_info['cpu_cores']} 线程)\n"
            f"内存: {sys_info['ram_gb']} GB\n"
            f"磁盘剩余: {sys_info['disk_free_gb']} GB"
        )

        # 摘要
        if missing:
            critical = [m for m in missing if m in ("PyTorch", "Ultralytics", "PySide6")]
            if critical:
                self.summary_label.setText(f"⚠ 缺少关键依赖：{', '.join(critical)}。请运行 pip install -r requirements.txt")
                self.summary_label.setObjectName("error")
            else:
                self.summary_label.setText(f"⚠ 缺少可选依赖：{', '.join(missing)}")
                self.summary_label.setObjectName("warning")
        elif not torch_info["available"]:
            self.summary_label.setText("✓ 依赖完整。当前无 CUDA GPU，训练将使用 CPU（速度较慢，建议在 AutoDL 等云 GPU 上训练）。")
            self.summary_label.setObjectName("warning")
        else:
            self.summary_label.setText("✓ 环境完全就绪，可使用 GPU 训练。")
            self.summary_label.setObjectName("success")
