#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
YOLO姿势标注可视化工具
用于查看和比较不同模型的标注结果质量
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import glob
from pathlib import Path
import re

# 确保可以导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from scripts.utils.visualization import AnnotationVisualizer
except ImportError:
    class AnnotationVisualizer:
        """简易版可视化器，用于绘制标注"""
        
        def __init__(self):
            self.colors = []  # 不同类别的颜色
            for i in range(10):  # 支持10种类别颜色
                color = (
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                )
                self.colors.append(color)
            # 添加关键点大小属性
            self.keypoint_size = 4
        
        def draw_annotation(self, image, class_idx, bbox, keypoints):
            """绘制标注"""
            h, w = image.shape[:2]
            color = self.colors[int(class_idx) % len(self.colors)]
            
            # 绘制边界框
            if bbox is not None:
                x_min, y_min, x_max, y_max = map(int, bbox)
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
                
                # 添加类别标签
                class_names = {
                    0: "correct_posture",
                    1: "bowed_head",
                    2: "desk_leaning",
                    3: "head_tilt",
                    4: "left_headed",
                    5: "right_headed",
                    6: "playing_object"
                }
                label = class_names.get(class_idx, f"class_{class_idx}")
                cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 绘制关键点
            if keypoints is not None:
                kp_size = getattr(self, 'keypoint_size', 4)  # 获取关键点大小，默认为4
                
                for i in range(0, len(keypoints), 3):
                    x, y, v = keypoints[i:i+3]
                    x, y = int(x), int(y)
                    
                    # 根据可见度决定绘制方式
                    if v >= 2:  # 可见
                        cv2.circle(image, (x, y), kp_size, color, -1)
                    elif v >= 1:  # 被遮挡
                        cv2.circle(image, (x, y), kp_size, color, 1)
                    else:  # 不可见
                        continue
                    
                    # 添加关键点索引
                    kp_idx = i // 3
                    cv2.putText(image, str(kp_idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 添加模型置信度信息（如果有）
            if hasattr(self, 'confidence') and self.confidence is not None:
                conf_text = f"Conf: {self.confidence:.2f}"
                cv2.putText(image, conf_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            return image

class YOLOAnnotationConverter:
    """YOLO标注格式转换器"""
    
    def __init__(self, num_keypoints=11):
        self.num_keypoints = num_keypoints
    
    def parse_txt_annotation(self, annotation_path, image_shape):
        """解析YOLO格式的txt标注文件"""
        h, w = image_shape[:2]
        
        try:
            with open(annotation_path, 'r') as f:
                line = f.readline().strip()
            
            parts = line.split()
            if len(parts) < 5 + self.num_keypoints * 3:
                raise ValueError(f"标注文件格式不正确: {annotation_path}")
            
            # 解析类别和边界框
            class_idx = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            
            # 将归一化坐标转回像素坐标
            x_center *= w
            y_center *= h
            width *= w
            height *= h
            
            # 计算边界框坐标
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            
            bbox = [x_min, y_min, x_max, y_max]
            
            # 解析关键点
            keypoints = []
            for i in range(self.num_keypoints):
                idx = 5 + i * 3
                if idx + 2 >= len(parts):
                    break
                    
                kp_x = float(parts[idx]) * w
                kp_y = float(parts[idx + 1]) * h
                kp_v = int(parts[idx + 2])
                
                keypoints.extend([kp_x, kp_y, kp_v])
            
            return class_idx, bbox, keypoints
            
        except Exception as e:
            print(f"解析标注文件出错: {e}")
            return None, None, None

class AnnotationViewer(tk.Tk):
    """标注可视化工具主窗口"""
    
    def __init__(self):
        super().__init__()
        
        self.title("YOLO姿势标注可视化工具")
        self.geometry("1400x800")
        
        # 初始化状态变量
        self.current_image_path = None
        self.current_annotation_path = None
        self.images_list = []
        self.current_image_index = 0
        self.available_models = []
        self.selected_models = []
        
        # 图像和标注处理器
        self.visualizer = AnnotationVisualizer()
        self.converter = YOLOAnnotationConverter()
        
        # 创建UI
        self._create_ui()
        
        # 绑定键盘事件
        self.bind("<Left>", lambda e: self._prev_image())
        self.bind("<Right>", lambda e: self._next_image())
        self.bind("<Up>", lambda e: self._update_keypoint_size(1))
        self.bind("<Down>", lambda e: self._update_keypoint_size(-1))
        
        # 配置变量
        self.keypoint_size = 4
    
    def _create_ui(self):
        """创建用户界面"""
        # 主窗口分为左右两部分
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧：控制面板
        control_frame = ttk.Frame(main_paned)
        main_paned.add(control_frame, weight=1)
        
        # 右侧：图像显示
        display_frame = ttk.Frame(main_paned)
        main_paned.add(display_frame, weight=5)
        
        # 设置左侧控制面板
        self._setup_control_panel(control_frame)
        
        # 设置右侧显示面板
        self._setup_display_panel(display_frame)
        
        # 设置最小宽度 - 修复方法错误，替换为适当的设置方式
        control_frame.config(width=250, height=600)
        control_frame.pack_propagate(False)
        
    def _setup_control_panel(self, parent):
        """设置控制面板"""
        # 数据集路径选择
        path_frame = ttk.LabelFrame(parent, text="数据集路径")
        path_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        self.dataset_path_var = tk.StringVar()
        
        ttk.Entry(path_frame, textvariable=self.dataset_path_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        ttk.Button(path_frame, text="浏览...", command=self._browse_dataset).pack(side=tk.RIGHT, padx=5, pady=5)
        
        # 图像列表
        images_frame = ttk.LabelFrame(parent, text="图像列表")
        images_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加搜索框
        search_frame = ttk.Frame(images_frame)
        search_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._filter_images)
        
        ttk.Label(search_frame, text="搜索:").pack(side=tk.LEFT, padx=2)
        ttk.Entry(search_frame, textvariable=self.search_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(search_frame, text="×", width=2, command=lambda: self.search_var.set("")).pack(side=tk.RIGHT, padx=2)
        
        # 图像列表显示
        list_frame = ttk.Frame(images_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.images_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.images_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.images_listbox.yview)
        
        self.images_listbox.bind("<<ListboxSelect>>", self._on_image_selected)
        
        # 模型选择
        models_frame = ttk.LabelFrame(parent, text="模型选择")
        models_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        self.models_var = tk.StringVar(value=[])
        self.models_listbox = tk.Listbox(models_frame, listvariable=self.models_var, selectmode=tk.MULTIPLE, height=6)
        self.models_listbox.pack(fill=tk.X, expand=True, padx=5, pady=5)
        self.models_listbox.bind("<<ListboxSelect>>", self._on_models_selected)
        
        # 控制按钮
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        ttk.Button(controls_frame, text="上一张", command=self._prev_image).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2, pady=5)
        ttk.Button(controls_frame, text="下一张", command=self._next_image).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2, pady=5)
        
        # 状态信息
        status_frame = ttk.LabelFrame(parent, text="状态信息")
        status_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="请选择数据集目录")
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=250).pack(fill=tk.X, expand=True, padx=5, pady=5)
        
    def _setup_display_panel(self, parent):
        """设置图像显示面板"""
        # 创建画布容器
        self.canvas_frame = ttk.Frame(parent)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建画布，用于显示图像
        self.canvas = tk.Canvas(self.canvas_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 信息面板
        info_frame = ttk.LabelFrame(parent, text="图像信息")
        info_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        # 显示当前图像信息
        self.info_text = tk.Text(info_frame, height=6, width=50, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建标签切换按钮
        toggle_frame = ttk.Frame(parent)
        toggle_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)
        
        self.show_labels_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggle_frame, text="显示标注", variable=self.show_labels_var, command=self._refresh_image).pack(side=tk.LEFT, padx=5)
        
        self.show_bbox_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggle_frame, text="显示边界框", variable=self.show_bbox_var, command=self._refresh_image).pack(side=tk.LEFT, padx=5)
        
        self.show_keypoints_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(toggle_frame, text="显示关键点", variable=self.show_keypoints_var, command=self._refresh_image).pack(side=tk.LEFT, padx=5)
        
        # 设置放大缩小按钮
        zoom_frame = ttk.Frame(toggle_frame)
        zoom_frame.pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(zoom_frame, text="-", width=2, command=lambda: self._update_keypoint_size(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Label(zoom_frame, text="关键点大小").pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="+", width=2, command=lambda: self._update_keypoint_size(1)).pack(side=tk.LEFT, padx=2)
    
    def _browse_dataset(self):
        """浏览选择数据集目录"""
        dataset_dir = filedialog.askdirectory(title="选择数据集目录")
        if not dataset_dir:
            return
            
        self.dataset_path_var.set(dataset_dir)
        self._load_dataset(dataset_dir)
    
    def _load_dataset(self, dataset_dir):
        """加载数据集"""
        try:
            # 查找所有图像文件
            image_extensions = ["*.jpg", "*.jpeg", "*.png"]
            images = []
            
            for ext in image_extensions:
                images.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))
            
            if not images:
                messagebox.showwarning("警告", f"在 {dataset_dir} 目录下未找到图像文件")
                return
            
            # 更新图像列表
            self.images_list = sorted(images)
            self._update_images_listbox()
            
            # 扫描可用的模型目录（查找对应的标注）
            self._scan_model_directories(dataset_dir)
            
            # 更新状态
            self.status_var.set(f"已加载 {len(self.images_list)} 张图像\n可用模型: {len(self.available_models)}")
            
            # 如果有图像，自动选择第一张
            if self.images_list:
                self.current_image_index = 0
                self.images_listbox.selection_clear(0, tk.END)
                self.images_listbox.selection_set(0)
                self._load_current_image()
        
        except Exception as e:
            messagebox.showerror("错误", f"加载数据集失败: {str(e)}")
    
    def _scan_model_directories(self, dataset_dir):
        """扫描可用的模型目录"""
        # 重置模型列表
        self.available_models = []
        
        # 查找潜在的标注目录
        dataset_path = Path(dataset_dir)
        parent_dir = dataset_path.parent
        
        # 1. 首先检查本目录下是否有labels目录
        if (dataset_path / "labels").exists():
            self.available_models.append(("Default", dataset_path / "labels"))
        
        # 2. 检查train/labels和val/labels目录
        if (dataset_path / "train" / "labels").exists():
            self.available_models.append(("Default-Train", dataset_path / "train" / "labels"))
        
        if (dataset_path / "val" / "labels").exists():
            self.available_models.append(("Default-Val", dataset_path / "val" / "labels"))
        
        # 3. 检查兄弟目录，查找可能的模型输出目录
        for sibling in parent_dir.iterdir():
            if sibling.is_dir() and sibling != dataset_path:
                # 检查是否有labels子目录
                if (sibling / "labels").exists():
                    model_name = sibling.name
                    self.available_models.append((model_name, sibling / "labels"))
                
                # 检查train/labels和val/labels子目录
                if (sibling / "train" / "labels").exists():
                    model_name = f"{sibling.name}-Train"
                    self.available_models.append((model_name, sibling / "train" / "labels"))
                
                if (sibling / "val" / "labels").exists():
                    model_name = f"{sibling.name}-Val"
                    self.available_models.append((model_name, sibling / "val" / "labels"))
        
        # 更新模型选择列表
        self.models_var.set([m[0] for m in self.available_models])
        
        # 默认选择第一个模型
        if self.available_models:
            self.models_listbox.selection_clear(0, tk.END)
            self.models_listbox.selection_set(0)
            self.selected_models = [self.available_models[0]]
    
    def _update_images_listbox(self):
        """更新图像列表框"""
        self.images_listbox.delete(0, tk.END)
        filter_text = self.search_var.get().lower()
        
        for img_path in self.images_list:
            if filter_text in os.path.basename(img_path).lower():
                self.images_listbox.insert(tk.END, os.path.basename(img_path))
    
    def _filter_images(self, *args):
        """根据搜索文本过滤图像列表"""
        self._update_images_listbox()
    
    def _on_image_selected(self, event):
        """当在列表中选择图像时触发"""
        selection = self.images_listbox.curselection()
        if not selection:
            return
        
        # 更新当前图像索引
        self.current_image_index = selection[0]
        self._load_current_image()
    
    def _on_models_selected(self, event):
        """当选择模型时触发"""
        selection = self.models_listbox.curselection()
        if not selection:
            return
            
        # 更新选择的模型
        self.selected_models = [self.available_models[idx] for idx in selection]
        
        # 重新加载当前图像的标注
        self._refresh_image()
    
    def _load_current_image(self):
        """加载当前选中的图像"""
        if not self.images_list or self.current_image_index >= len(self.images_list):
            return
        
        # 获取当前图像路径
        image_path = self.images_list[self.current_image_index]
        self.current_image_path = image_path
        
        # 更新UI
        self.images_listbox.see(self.current_image_index)
        self.images_listbox.selection_clear(0, tk.END)
        self.images_listbox.selection_set(self.current_image_index)
        
        # 加载图像
        self._refresh_image()
    
    def _refresh_image(self):
        """刷新显示当前图像和标注"""
        if not self.current_image_path:
            return
            
        try:
            # 读取原始图像
            original_image = cv2.imread(self.current_image_path)
            if original_image is None:
                raise ValueError(f"无法读取图像: {self.current_image_path}")
            
            # 转换为RGB（PIL使用RGB）
            image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            # 为每个选择的模型绘制标注
            annotated_images = []
            
            # 如果没有选择任何模型，至少显示原始图像
            if not self.selected_models:
                annotated_images.append(("原始图像", image_rgb.copy()))
            else:
                for model_name, labels_dir in self.selected_models:
                    # 创建图像副本
                    annotated_image = image_rgb.copy()
                    
                    # 查找对应的标注文件
                    image_filename = os.path.basename(self.current_image_path)
                    label_filename = os.path.splitext(image_filename)[0] + ".txt"
                    label_path = os.path.join(labels_dir, label_filename)
                    
                    # 如果标注文件存在，加载并绘制
                    if os.path.exists(label_path):
                        # 解析标注
                        class_idx, bbox, keypoints = self.converter.parse_txt_annotation(label_path, annotated_image.shape)
                        
                        if class_idx is not None:
                            # 根据用户选择决定显示内容
                            draw_bbox = bbox if self.show_bbox_var.get() else None
                            draw_keypoints = keypoints if self.show_keypoints_var.get() else None
                            
                            if self.show_labels_var.get() and (draw_bbox is not None or draw_keypoints is not None):
                                # 设置关键点大小
                                self.visualizer.keypoint_size = self.keypoint_size
                                # 绘制标注
                                annotated_image = self.visualizer.draw_annotation(
                                    cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR), 
                                    class_idx, 
                                    draw_bbox, 
                                    draw_keypoints
                                )
                                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    
                    # 添加模型名称水印
                    cv2.putText(
                        annotated_image, 
                        model_name, 
                        (10, h - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, 
                        (255, 0, 0), 
                        2
                    )
                    
                    annotated_images.append((model_name, annotated_image))
            
            # 组合所有标注图像
            if not annotated_images:
                # 如果没有任何图像，显示原始图像
                display_image = image_rgb
            elif len(annotated_images) == 1:
                # 单个模型，直接显示
                display_image = annotated_images[0][1]
            else:
                # 多个模型，横向或纵向拼接
                if len(annotated_images) <= 3:
                    # 横向拼接（最多3个）
                    display_image = np.hstack([img for _, img in annotated_images])
                else:
                    # 网格排列
                    cols = min(3, len(annotated_images))
                    rows = (len(annotated_images) + cols - 1) // cols
                    
                    # 创建网格
                    grid_images = []
                    for i in range(rows):
                        row_images = []
                        for j in range(cols):
                            idx = i * cols + j
                            if idx < len(annotated_images):
                                row_images.append(annotated_images[idx][1])
                            else:
                                # 填充空白图像
                                row_images.append(np.zeros_like(annotated_images[0][1]))
                        grid_images.append(np.hstack(row_images))
                    
                    display_image = np.vstack(grid_images)
            
            # 调整显示图像尺寸以适应窗口
            self._display_image(display_image)
            
            # 更新信息面板
            self._update_info_panel()
            
            # 更新状态
            img_name = os.path.basename(self.current_image_path)
            self.status_var.set(f"当前图像: {img_name}\n索引: {self.current_image_index+1}/{len(self.images_list)}")
        
        except Exception as e:
            messagebox.showerror("错误", f"加载图像失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _display_image(self, image):
        """在画布上显示图像"""
        # 获取画布尺寸
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # 如果画布尚未配置好尺寸，使用默认值
        if canvas_width <= 1:
            canvas_width = 800
        if canvas_height <= 1:
            canvas_height = 600
        
        # 计算图像缩放比例
        h, w = image.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        
        # 如果图像太大，进行缩放
        if scale < 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 转换为PIL图像，然后转换为PhotoImage
        pil_image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(pil_image)
        
        # 清除画布并显示新图像
        self.canvas.delete("all")
        self.canvas.config(width=photo.width(), height=photo.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        
        # 保存引用以避免被垃圾回收
        self.current_photo = photo
    
    def _update_info_panel(self):
        """更新信息面板内容"""
        if not self.current_image_path:
            return
        
        # 清除现有内容
        self.info_text.delete(1.0, tk.END)
        
        # 添加图像基本信息
        image_name = os.path.basename(self.current_image_path)
        image_size = os.path.getsize(self.current_image_path) / 1024  # KB
        
        img = cv2.imread(self.current_image_path)
        h, w = img.shape[:2]
        
        self.info_text.insert(tk.END, f"文件名: {image_name}\n")
        self.info_text.insert(tk.END, f"尺寸: {w}x{h}\n")
        self.info_text.insert(tk.END, f"大小: {image_size:.1f} KB\n")
        
        # 添加标注信息
        for model_name, labels_dir in self.selected_models:
            # 查找对应的标注文件
            label_filename = os.path.splitext(image_name)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_filename)
            
            if os.path.exists(label_path):
                # 读取标注内容
                with open(label_path, 'r') as f:
                    label_content = f.read().strip()
                
                # 简单解析并格式化
                class_idx = label_content.split()[0]
                num_keypoints = (len(label_content.split()) - 5) // 3
                
                self.info_text.insert(tk.END, f"\n{model_name}标注:\n")
                self.info_text.insert(tk.END, f"类别: {class_idx}\n")
                self.info_text.insert(tk.END, f"关键点数: {num_keypoints}\n")
            else:
                self.info_text.insert(tk.END, f"\n{model_name}标注: 未找到\n")
    
    def _prev_image(self):
        """显示上一张图像"""
        if not self.images_list:
            return
            
        # 更新索引
        self.current_image_index = (self.current_image_index - 1) % len(self.images_list)
        self._load_current_image()
    
    def _next_image(self):
        """显示下一张图像"""
        if not self.images_list:
            return
            
        # 更新索引
        self.current_image_index = (self.current_image_index + 1) % len(self.images_list)
        self._load_current_image()
    
    def _update_keypoint_size(self, delta):
        """更新关键点显示大小"""
        self.keypoint_size = max(1, self.keypoint_size + delta)
        self._refresh_image()

def main():
    """启动应用"""
    try:
        # 尝试导入并使用主题
        from ttkthemes import ThemedStyle
        app = AnnotationViewer()
        style = ThemedStyle(app)
        style.set_theme("arc")  # 设置一个现代主题
    except ImportError:
        # 如果导入失败，使用默认样式
        app = AnnotationViewer()
    
    # 启动应用
    app.mainloop()

if __name__ == "__main__":
    main() 