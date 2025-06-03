#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版YOLO姿势标注可视化工具
用于查看标注结果质量和对比不同置信度下的标注效果
"""

import os
import sys
import argparse
import glob
from pathlib import Path
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# 添加项目根目录到路径以导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from scripts.utils.visualization import AnnotationVisualizer
except ImportError:
    print("无法导入AnnotationVisualizer，使用内置的简化版可视化器")
    
    class AnnotationVisualizer:
        """简化版标注可视化工具"""
        
        def __init__(self, keypoint_radius=5, line_thickness=2):
            self.keypoint_radius = keypoint_radius
            self.line_thickness = line_thickness
            
            # 定义类别颜色
            self.class_colors = {
                0: (0, 255, 0),    # 正确姿势 - 绿色
                1: (0, 0, 255),    # 低头姿势 - 红色
                2: (255, 0, 0),    # 趴桌姿势 - 蓝色
                3: (255, 255, 0),  # 头部倾斜 - 青色
                4: (255, 0, 255),  # 左偏头部 - 紫色
                5: (0, 255, 255),  # 右偏头部 - 黄色
                6: (128, 128, 128) # 玩物姿势 - 灰色
            }
            
            # 骨架连接
            self.skeleton = [
                (0, 1),  # 鼻子 - 左眼
                (0, 2),  # 鼻子 - 右眼
                (1, 3),  # 左眼 - 左耳
                (2, 4),  # 右眼 - 右耳
                (5, 6),  # 左肩 - 右肩
                (5, 7),  # 左肩 - 左肘
                (6, 8),  # 右肩 - 右肘
                (7, 9),  # 左肘 - 左手腕
                (8, 10)  # 右肘 - 右手腕
            ]
        
        def draw_annotation(self, image, class_idx, bbox, keypoints):
            """在图像上绘制标注"""
            # 复制图像以避免修改原始图像
            result = image.copy()
            
            # 获取类别颜色
            color = self.class_colors.get(class_idx, (0, 255, 0))
            
            # 绘制边界框
            if bbox is not None:
                x_min, y_min, x_max, y_max = map(int, bbox)
                cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color, self.line_thickness)
                
                # 绘制类别标签
                class_names = {
                    0: "correct_posture",
                    1: "bowed_head",
                    2: "desk_leaning",
                    3: "head_tilt",
                    4: "left_headed",
                    5: "right_headed",
                    6: "playing_object"
                }
                label = class_names.get(class_idx, f"Class {class_idx}")
                cv2.putText(result, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, color, self.line_thickness)
            
            # 绘制关键点
            if keypoints is not None:
                # 提取关键点为列表
                points = []
                for i in range(0, len(keypoints), 3):
                    if i + 2 >= len(keypoints):
                        break
                    x, y, v = keypoints[i:i+3]
                    points.append((int(x), int(y), v))
                
                # 绘制关键点
                for i, (x, y, v) in enumerate(points):
                    if v > 0:  # 仅绘制可见的关键点
                        # 可见性等级影响绘制样式
                        if v >= 2:  # 高置信度
                            cv2.circle(result, (x, y), self.keypoint_radius, color, -1)  # 填充
                        else:  # 低置信度
                            cv2.circle(result, (x, y), self.keypoint_radius, color, 1)   # 轮廓
                        
                        # 添加关键点索引标签
                        cv2.putText(result, str(i), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.4, color, 1)
                
                # 绘制骨架线条
                for connection in self.skeleton:
                    if max(connection) >= len(points):
                        continue
                    
                    pt1_idx, pt2_idx = connection
                    pt1 = points[pt1_idx]
                    pt2 = points[pt2_idx]
                    
                    # 仅当两个点都可见时绘制连线
                    if pt1[2] > 0 and pt2[2] > 0:
                        cv2.line(result, (pt1[0], pt1[1]), (pt2[0], pt2[1]), 
                                 color, max(1, self.line_thickness - 1))
            
            return result


class YOLOAnnotationViewer:
    """YOLO标注查看器"""
    
    def __init__(self, dataset_dir=None):
        """初始化查看器"""
        self.dataset_dir = Path(dataset_dir) if dataset_dir else None
        self.visualizer = AnnotationVisualizer()
        
        # 图像和标注数据
        self.current_image_path = None
        self.current_image = None
        self.images_list = []
        self.current_index = 0
        
        # 初始化UI
        self._init_ui()
        
        # 加载数据集（如果指定）
        if self.dataset_dir:
            self._load_dataset()
    
    def _init_ui(self):
        """初始化用户界面"""
        self.window = tk.Tk()
        self.window.title("YOLO标注查看器")
        self.window.geometry("1200x800")
        
        # 创建菜单
        menubar = tk.Menu(self.window)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="打开数据集", command=self._browse_dataset)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.window.quit)
        
        menubar.add_cascade(label="文件", menu=file_menu)
        self.window.config(menu=menubar)
        
        # 主界面分为左右两部分
        main_pane = ttk.PanedWindow(self.window, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧：图像列表和控制面板
        left_frame = ttk.Frame(main_pane, width=250)
        main_pane.add(left_frame, weight=1)
        
        # 右侧：图像显示
        right_frame = ttk.Frame(main_pane)
        main_pane.add(right_frame, weight=4)
        
        # 设置左侧控制面板
        self._setup_control_panel(left_frame)
        
        # 设置右侧图像显示
        self._setup_image_panel(right_frame)
        
        # 绑定键盘事件
        self.window.bind("<Left>", lambda e: self._prev_image())
        self.window.bind("<Right>", lambda e: self._next_image())
        self.window.bind("<Up>", lambda e: self._adjust_keypoint_size(1))
        self.window.bind("<Down>", lambda e: self._adjust_keypoint_size(-1))
    
    def _setup_control_panel(self, parent):
        """设置控制面板"""
        # 数据集路径显示
        path_frame = ttk.LabelFrame(parent, text="数据集路径")
        path_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(path_frame, textvariable=self.path_var, state="readonly")
        path_entry.pack(fill=tk.X, padx=5, pady=5)
        
        # 图像列表
        list_frame = ttk.LabelFrame(parent, text="图像列表")
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 搜索框
        search_frame = ttk.Frame(list_frame)
        search_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(search_frame, text="搜索:").pack(side=tk.LEFT)
        
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self._filter_images)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # 图像列表显示
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set)
        self.listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scrollbar.config(command=self.listbox.yview)
        
        self.listbox.bind("<<ListboxSelect>>", self._on_image_selected)
        
        # 按钮控制
        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(buttons_frame, text="上一张", command=self._prev_image).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(buttons_frame, text="下一张", command=self._next_image).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        
        # 信息显示
        info_frame = ttk.LabelFrame(parent, text="图像信息")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_var = tk.StringVar()
        info_label = ttk.Label(info_frame, textvariable=self.info_var, wraplength=250)
        info_label.pack(fill=tk.X, padx=5, pady=5)
    
    def _setup_image_panel(self, parent):
        """设置图像显示面板"""
        # 图像显示
        self.canvas = tk.Canvas(parent, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 添加滚动条
        x_scrollbar = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.canvas.xview)
        y_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.canvas.yview)
        
        self.canvas.config(xscrollcommand=x_scrollbar.set, yscrollcommand=y_scrollbar.set)
        
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定鼠标滚轮缩放
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Control-MouseWheel>", self._on_zoom)
        
        # 图像缩放因子
        self.zoom_factor = 1.0
    
    def _browse_dataset(self):
        """浏览选择数据集目录"""
        folder = filedialog.askdirectory(title="选择数据集目录")
        if folder:
            self.dataset_dir = Path(folder)
            self.path_var.set(str(self.dataset_dir))
            self._load_dataset()
    
    def _load_dataset(self):
        """加载数据集图像和标注"""
        if not self.dataset_dir:
            return
        
        # 查找所有图像
        images = []
        
        # 检查训练集图像
        train_img_dir = self.dataset_dir / "train" / "images"
        if train_img_dir.exists():
            images.extend(list(train_img_dir.glob("*.jpg")) + 
                        list(train_img_dir.glob("*.jpeg")) + 
                        list(train_img_dir.glob("*.png")))
        
        # 检查验证集图像
        val_img_dir = self.dataset_dir / "val" / "images"
        if val_img_dir.exists():
            images.extend(list(val_img_dir.glob("*.jpg")) + 
                        list(val_img_dir.glob("*.jpeg")) + 
                        list(val_img_dir.glob("*.png")))
        
        if not images:
            messagebox.showwarning("警告", "未找到图像文件！")
            return
        
        # 更新图像列表
        self.images_list = sorted(images)
        self._update_listbox()
        
        # 加载第一张图像
        if self.images_list:
            self.current_index = 0
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(0)
            self.listbox.see(0)
            self._load_current_image()
    
    def _update_listbox(self):
        """更新列表框显示"""
        self.listbox.delete(0, tk.END)
        
        for img_path in self.images_list:
            # 提取文件名和是训练集还是验证集
            parts = str(img_path).split(os.sep)
            split = parts[-3]  # train 或 val
            filename = parts[-1]
            self.listbox.insert(tk.END, f"[{split}] {filename}")
    
    def _filter_images(self, *args):
        """根据搜索词筛选图像"""
        search_term = self.search_var.get().lower()
        
        if not search_term:
            # 显示所有图像
            self._update_listbox()
            return
        
        # 筛选匹配的图像
        self.listbox.delete(0, tk.END)
        
        for img_path in self.images_list:
            # 提取文件名和类别目录
            parts = str(img_path).split(os.sep)
            split = parts[-3]  # train 或 val
            filename = parts[-1]
            
            # 如果文件名包含搜索词，则添加到列表
            if search_term in filename.lower() or search_term in split.lower():
                self.listbox.insert(tk.END, f"[{split}] {filename}")
    
    def _on_image_selected(self, event):
        """图像选择事件处理"""
        selection = self.listbox.curselection()
        if not selection:
            return
        
        # 获取选择的索引
        index = selection[0]
        list_size = self.listbox.size()
        
        # 计算实际索引（考虑搜索筛选）
        if self.search_var.get():
            # 在搜索模式下，需要找到对应的真实索引
            selected_text = self.listbox.get(index)
            for i, img_path in enumerate(self.images_list):
                parts = str(img_path).split(os.sep)
                split = parts[-3]
                filename = parts[-1]
                if selected_text == f"[{split}] {filename}":
                    self.current_index = i
                    break
        else:
            # 非搜索模式，直接使用索引
            self.current_index = index
        
        self._load_current_image()
    
    def _load_current_image(self):
        """加载当前选中的图像和标注"""
        if not self.images_list or self.current_index >= len(self.images_list):
            return
        
        # 获取图像路径
        img_path = self.images_list[self.current_index]
        self.current_image_path = img_path
        
        # 找到对应的标注文件
        parts = str(img_path).split(os.sep)
        split = parts[-3]  # train 或 val
        img_name = parts[-1]
        
        label_dir = self.dataset_dir / split / "labels"
        label_path = label_dir / (Path(img_name).stem + ".txt")
        
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            messagebox.showerror("错误", f"无法读取图像: {img_path}")
            return
        
        # 保存原始图像
        self.current_image = img.copy()
        
        # 如果有标注文件，加载并标注
        annotated_img = img.copy()
        
        if label_path.exists():
            # 解析标注文件
            class_idx, bbox, keypoints = self._parse_yolo_annotation(label_path, img.shape)
            
            if class_idx is not None:
                # 使用可视化器绘制标注
                annotated_img = self.visualizer.draw_annotation(img, class_idx, bbox, keypoints)
                
                # 更新信息面板
                self._update_info(img_path, class_idx, bbox, keypoints)
        else:
            self._update_info(img_path, None, None, None)
        
        # 显示图像
        self._display_image(annotated_img)
    
    def _parse_yolo_annotation(self, label_path, image_shape):
        """解析YOLO格式的标注文件"""
        h, w = image_shape[:2]
        
        try:
            with open(label_path, 'r') as f:
                line = f.readline().strip()
            
            if not line:
                return None, None, None
            
            parts = line.split()
            if len(parts) < 5:
                return None, None, None
            
            # 解析类别和边界框
            class_idx = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            
            # 转换坐标（从归一化坐标到像素坐标）
            x_center *= w
            y_center *= h
            width *= w
            height *= h
            
            # 计算边界框顶点坐标
            x_min = max(0, int(x_center - width / 2))
            y_min = max(0, int(y_center - height / 2))
            x_max = min(w, int(x_center + width / 2))
            y_max = min(h, int(y_center + height / 2))
            
            bbox = [x_min, y_min, x_max, y_max]
            
            # 解析关键点
            keypoints = []
            for i in range(5, len(parts), 3):
                if i + 2 >= len(parts):
                    break
                
                kp_x = float(parts[i]) * w
                kp_y = float(parts[i + 1]) * h
                kp_v = float(parts[i + 2])
                
                keypoints.extend([kp_x, kp_y, kp_v])
            
            return class_idx, bbox, keypoints
            
        except Exception as e:
            print(f"解析标注文件出错: {e}")
            return None, None, None
    
    def _display_image(self, image):
        """在Canvas上显示图像"""
        h, w = image.shape[:2]
        
        # 将OpenCV图像转换为PIL格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # 应用缩放
        if self.zoom_factor != 1.0:
            new_w = int(w * self.zoom_factor)
            new_h = int(h * self.zoom_factor)
            pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
        
        # 转换为Tkinter图像对象
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        # 更新Canvas
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        # 配置滚动区域
        self.canvas.config(scrollregion=(0, 0, self.tk_image.width(), self.tk_image.height()))
    
    def _update_info(self, img_path, class_idx, bbox, keypoints):
        """更新信息面板"""
        info_text = f"文件: {Path(img_path).name}\n"
        
        if class_idx is not None:
            # 类别名称
            class_names = {
                0: "correct_posture",
                1: "bowed_head",
                2: "desk_leaning",
                3: "head_tilt",
                4: "left_headed",
                5: "right_headed",
                6: "playing_object"
            }
            class_name = class_names.get(class_idx, f"类别{class_idx}")
            info_text += f"类别: {class_name}\n"
            
            # 计算可见关键点数量
            if keypoints:
                visible_keypoints = sum(1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0)
                total_keypoints = len(keypoints) // 3
                info_text += f"关键点: {visible_keypoints}/{total_keypoints} 可见\n"
        else:
            info_text += "未找到标注"
        
        self.info_var.set(info_text)
    
    def _prev_image(self):
        """显示上一张图像"""
        if not self.images_list:
            return
        
        self.current_index = max(0, self.current_index - 1)
        self._load_current_image()
        
        # 更新列表框选择
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.current_index)
        self.listbox.see(self.current_index)
    
    def _next_image(self):
        """显示下一张图像"""
        if not self.images_list:
            return
        
        self.current_index = min(len(self.images_list) - 1, self.current_index + 1)
        self._load_current_image()
        
        # 更新列表框选择
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.current_index)
        self.listbox.see(self.current_index)
    
    def _on_mousewheel(self, event):
        """鼠标滚轮事件处理（滚动）"""
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _on_zoom(self, event):
        """Ctrl+滚轮缩放图像"""
        if event.delta > 0:
            # 放大
            self.zoom_factor *= 1.1
        else:
            # 缩小
            self.zoom_factor /= 1.1
        
        # 限制缩放范围
        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))
        
        # 重新显示当前图像
        if self.current_image is not None:
            img = self.current_image.copy()
            
            # 找到标注信息
            if self.current_image_path:
                parts = str(self.current_image_path).split(os.sep)
                split = parts[-3]
                img_name = parts[-1]
                
                label_dir = self.dataset_dir / split / "labels"
                label_path = label_dir / (Path(img_name).stem + ".txt")
                
                if label_path.exists():
                    class_idx, bbox, keypoints = self._parse_yolo_annotation(label_path, img.shape)
                    if class_idx is not None:
                        img = self.visualizer.draw_annotation(img, class_idx, bbox, keypoints)
            
            self._display_image(img)
    
    def _adjust_keypoint_size(self, delta):
        """调整关键点大小"""
        new_size = self.visualizer.keypoint_radius + delta
        self.visualizer.keypoint_radius = max(1, min(20, new_size))
        
        # 重新加载当前图像以应用新的关键点大小
        self._load_current_image()
    
    def run(self):
        """运行查看器"""
        self.window.mainloop()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLO标注可视化工具')
    parser.add_argument('--dataset', type=str, default=None, 
                        help='数据集目录路径')
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    viewer = YOLOAnnotationViewer(args.dataset)
    viewer.run()

if __name__ == "__main__":
    main() 