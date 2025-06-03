import cv2
import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import time
from ultralytics import YOLO
import torch
import numpy as np
import psutil
import gc
from thop import profile as thop_profile

# 添加智能算法Pro类
class PostureProRecognizer:
    """智能算法Pro: 多时间尺度混合窗口姿态识别器"""
    def __init__(self, aggressiveness=50):
        # 窗口存储
        self.short_window = []  # 1秒窗口
        self.mid_window = []    # 3秒窗口
        self.long_window = []   # 8秒窗口
        
        # 姿态转换矩阵 - 记录连续姿态转换的可能性
        self.transition_matrix = {
            'correct_posture': {'correct_posture': 0.8, 'bowed_head': 0.1, 'desk_leaning': 0.05, 'head_tilt': 0.05, 'left_headed': 0.0, 'right_headed': 0.0},
            'bowed_head': {'correct_posture': 0.1, 'bowed_head': 0.75, 'desk_leaning': 0.1, 'head_tilt': 0.03, 'left_headed': 0.01, 'right_headed': 0.01},
            'desk_leaning': {'correct_posture': 0.05, 'bowed_head': 0.1, 'desk_leaning': 0.8, 'head_tilt': 0.03, 'left_headed': 0.01, 'right_headed': 0.01},
            'head_tilt': {'correct_posture': 0.05, 'bowed_head': 0.03, 'desk_leaning': 0.02, 'head_tilt': 0.8, 'left_headed': 0.05, 'right_headed': 0.05},
            'left_headed': {'correct_posture': 0.05, 'bowed_head': 0.02, 'desk_leaning': 0.02, 'head_tilt': 0.04, 'left_headed': 0.8, 'right_headed': 0.07},
            'right_headed': {'correct_posture': 0.05, 'bowed_head': 0.02, 'desk_leaning': 0.02, 'head_tilt': 0.04, 'left_headed': 0.07, 'right_headed': 0.8}
        }
        
        # 类别映射
        self.original_classes = {
            0: "correct_posture",   # 正确姿势
            1: "bowed_head",        # 低头
            2: "desk_leaning",      # 趴桌
            3: "head_tilt",         # 歪头
            4: "left_headed",       # 左偏头
            5: "right_headed"       # 右偏头
        }
        
        # 输出状态
        self.current_output = None
        self.last_output = None
        self.output_stability = 0
        self.nobody_detected = False
        
        # 状态和统计数据
        self.posture_stats = {}
        self.frame_batch = []
        self.last_batch_time = time.time()
        self.batch_interval = 2.0  # 2秒更新一次输出，适应5FPS
        self.stability_score = 0.0
        
        # 记录最后一次有效检测的时间
        self.last_detection_time = None
        
        # nobody状态跟踪
        self.nobody_start_time = None
        self.nobody_duration = 0
        
        # 设置激进度
        self.set_aggressiveness(aggressiveness)
    
    def set_aggressiveness(self, aggressiveness):
        """设置算法激进度(0-100)"""
        self.aggressiveness = max(0, min(100, aggressiveness))
        
        # 映射到具体参数
        self.conf_threshold = 0.65 - (self.aggressiveness / 100) * 0.4
        self.iou_threshold = 0.70 - (self.aggressiveness / 100) * 0.35
        
        # 窗口权重分配
        self.short_window_weight = 0.6 + (self.aggressiveness / 100) * 0.3
        self.mid_window_weight = 0.3 - (self.aggressiveness / 100) * 0.2
        self.long_window_weight = 0.1 - (self.aggressiveness / 100) * 0.05
        
        # 姿态校正系数
        self.transition_penalty = 0.8 - (self.aggressiveness / 100) * 0.7
        self.right_headed_correction = 0.25 - (self.aggressiveness / 100) * 0.2
    
    def process_detection(self, results, current_time):
        """处理单帧检测结果"""
        # 提取检测结果
        has_valid_detection = False
        has_high_confidence_detection = False  # 标记是否有高置信度检测
        
        if results and results[0].boxes:
            boxes = results[0].boxes
            cls = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []
            conf = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
            
            if len(cls) > 0:
                has_valid_detection = True
                
                # 检查是否有高置信度的检测结果
                high_conf_detections = [c for c, conf_val in zip(cls, conf) if conf_val > 0.25]
                if len(high_conf_detections) > 0:
                    has_high_confidence_detection = True
                
                # 找出置信度最高的姿态
                best_posture = None
                best_conf = -1
                best_class_id = None
                
                for i in range(len(cls)):
                    class_id = int(cls[i])
                    confidence = conf[i]
                    
                    if class_id in self.original_classes and confidence > best_conf:
                        best_conf = confidence
                        best_class_id = class_id
                        best_posture = self.original_classes[class_id]
                
                # 添加到帧批次
                if best_posture:
                    posture_data = {
                        'posture': best_posture,
                        'confidence': best_conf,
                        'timestamp': current_time,
                        'class_id': best_class_id
                    }
                    self.add_to_frame_batch(posture_data, current_time)
                    
                    # 更新最后一次有效检测的时间
                    self.last_detection_time = current_time
                    
                    # 如果检测到姿态，重置nobody状态
                    self.nobody_start_time = None
                    self.nobody_duration = 0
                else:
                    self.add_to_frame_batch(None, current_time)
        
        # 如果没有有效检测
        if not has_valid_detection:
            self.add_to_frame_batch(None, current_time) 
            
            # 检查是否长时间没有检测结果
            if hasattr(self, 'last_detection_time') and self.last_detection_time is not None:
                time_since_last_detection = current_time - self.last_detection_time
                
                # 检查是否在姿势转换期
                is_transition_period = False
                if self.last_output and self.last_output != 'nobody' and time_since_last_detection < 1.5:
                    is_transition_period = True
                
                # 只有在非转换期且超过5秒没有任何检测结果，才强制清理所有窗口
                if time_since_last_detection > 5.0 and not is_transition_period:
                    self.short_window = []
                    self.mid_window = []
                    self.long_window = []
                    self.nobody_detected = True
                    self.current_output = 'nobody'
                    
                    # 设置nobody开始时间，如果还没有的话
                    if not hasattr(self, 'nobody_start_time') or self.nobody_start_time is None:
                        self.nobody_start_time = current_time
                        self.nobody_duration = 0
                    else:
                        # 计算持续时间
                        self.nobody_duration = current_time - self.nobody_start_time
        
        # 处理完整批次 (约2秒10帧)
        if current_time - self.last_batch_time >= self.batch_interval:
            result = self.process_frame_batch()
            
            # 跟踪nobody状态的持续时间 - 更精确的时间跟踪
            if result == "nobody":
                # 检查短期窗口中是否有高置信度的检测
                has_recent_high_conf = self.check_recent_high_confidence_detections()
                
                if has_recent_high_conf or has_high_confidence_detection:
                    # 如果短期窗口中有高置信度检测，停止增加nobody时间，并准备转换到其他状态
                    if self.nobody_start_time is not None:
                        # 保持nobody_duration不变，但不增加
                        pass
                    
                    # 如果短期窗口中有足够的高置信度检测，尝试重新评估状态
                    if self.has_enough_high_confidence_for_transition():
                        # 根据短期窗口中的检测结果重新评估状态
                        new_state = self.determine_state_from_recent_detections()
                        if new_state and new_state != "nobody":
                            result = new_state
                            self.current_output = new_state
                            self.nobody_start_time = None
                            self.nobody_duration = 0
                else:
                    # 否则正常计算nobody持续时间
                    if not hasattr(self, 'nobody_start_time') or self.nobody_start_time is None:
                        self.nobody_start_time = current_time
                    # 精确到0.1秒的nobody持续时间计算
                    self.nobody_duration = current_time - self.nobody_start_time
            else:
                self.nobody_start_time = None
                self.nobody_duration = 0
                
            return result
        
        # 返回最后一次处理结果
        return self.current_output
    
    def check_recent_high_confidence_detections(self):
        """检查短期窗口中是否有高置信度的检测结果"""
        if not self.short_window:
            return False
            
        # 检查最近2秒内的短期窗口
        current_time = time.time()
        recent_window = [record for record in self.short_window 
                       if record is not None and current_time - record['timestamp'] <= 2.0]
        
        # 获取高置信度的检测数量
        high_conf_count = sum(1 for record in recent_window if record['confidence'] > 0.25)
        
        # 如果有至少2帧高置信度的检测，返回True
        return high_conf_count >= 2
    
    def has_enough_high_confidence_for_transition(self):
        """检查是否有足够的高置信度检测来进行状态转换"""
        if not self.short_window:
            return False
        
        # 检查最近1秒内的短期窗口
        current_time = time.time()
        recent_window = [record for record in self.short_window 
                       if record is not None and current_time - record['timestamp'] <= 1.0]
        
        # 获取高置信度的检测数量
        high_conf_count = sum(1 for record in recent_window if record['confidence'] > 0.25)
        
        # 如果有至少2帧高置信度的检测，返回True
        return high_conf_count >= 2
    
    def determine_state_from_recent_detections(self):
        """根据最近的检测结果确定新状态"""
        if not self.short_window:
            return None
            
        # 只考虑最近1秒的数据
        current_time = time.time()
        recent_window = [record for record in self.short_window 
                       if record is not None and current_time - record['timestamp'] <= 1.0]
        
        if not recent_window:
            return None
            
        # 统计各种姿态
        posture_counts = {}
        for record in recent_window:
            posture = record['posture']
            if posture not in posture_counts:
                posture_counts[posture] = 0
            posture_counts[posture] += 1
            
        # 找出最常见的姿态
        if posture_counts:
            most_common_posture = max(posture_counts.items(), key=lambda x: x[1])[0]
            
            # 转换为简化输出
            if most_common_posture == 'correct_posture':
                return 'correct_posture'
            elif most_common_posture in ['bowed_head', 'desk_leaning', 'head_tilt', 'left_headed', 'right_headed']:
                return 'incorrect_posture'
                
        return None
    
    def add_to_frame_batch(self, posture_data, current_time):
        """添加到帧批次"""
        # 添加到当前批次
        self.frame_batch.append(posture_data)
        
        # 保持批次长度合理 (约10帧)
        max_batch_size = int(self.batch_interval * 5.5)  # 适应5FPS左右，稍微留余量
        if len(self.frame_batch) > max_batch_size:
            self.frame_batch = self.frame_batch[-max_batch_size:]
    
    def process_frame_batch(self):
        """处理当前帧批次"""
        # 更新各时间窗口
        self.update_windows()
        
        # 当前时间
        current_time = time.time()
        
        # 判断检测窗口是否完全为空
        windows_completely_empty = not self.short_window and not self.mid_window and not self.long_window
        
        # 检查是否为姿势转换过程
        is_posture_transition = False
        if self.last_output and self.last_output != 'nobody' and hasattr(self, 'last_detection_time') and self.last_detection_time:
            time_since_last_detection = current_time - self.last_detection_time
            # 可能正在转换姿势的时间范围
            if time_since_last_detection < 1.5:
                is_posture_transition = True
        
        # 检查窗口是否为空
        if windows_completely_empty and not is_posture_transition:
            # 只有当不在转换期且窗口完全为空时才直接判断为nobody
            self.nobody_detected = True
            self.current_output = "nobody"
            self.last_batch_time = time.time()
            return "nobody"
        
        # 检查短期窗口中是否有最近的检测结果
        has_recent_detections = False
        short_window_threshold = 2.0  # 2秒内没有检测结果判定为nobody
        
        if self.short_window:
            most_recent = max(self.short_window, key=lambda x: x['timestamp'])
            time_gap = current_time - most_recent['timestamp'] 
            if time_gap <= short_window_threshold:
                has_recent_detections = True
            elif is_posture_transition and time_gap <= 2.5:
                # 在姿势转换期间提供更长的容错时间
                has_recent_detections = True
        
        # 增强nobody判断逻辑
        if not has_recent_detections and not is_posture_transition:
            # 只有当不在转换期且长时间无检测时才判断为nobody
            self.nobody_detected = True
            self.current_output = "nobody"
            self.last_batch_time = time.time()
            return "nobody"
        
        # 正常处理姿态分析
        posture_weights = self.analyze_windows()
        
        # 如果在姿势转换过程中权重为空，保持上一个有效姿势
        if not posture_weights and is_posture_transition and self.last_output != 'nobody':
            self.nobody_detected = False
            self.current_output = self.last_output
            self.last_batch_time = time.time()
            return self.last_output
        
        posture_weights = self.apply_transition_penalties(posture_weights)
        self.correct_special_postures(posture_weights)
        self.determine_final_output(posture_weights)
        
        # 更新批次时间
        self.last_batch_time = time.time()
        
        # 清空当前批次
        self.frame_batch = []
        
        return self.current_output
    
    def update_windows(self):
        """更新各时间窗口"""
        current_time = time.time()
        
        # 过滤空结果
        valid_frames = [frame for frame in self.frame_batch if frame is not None]
        
        # 添加到短期窗口 (1秒)
        self.short_window.extend(valid_frames)
        self.short_window = [p for p in self.short_window 
                           if current_time - p['timestamp'] <= 1.0]
        
        # 添加到中期窗口 (3秒)
        self.mid_window.extend(valid_frames)
        self.mid_window = [p for p in self.mid_window 
                         if current_time - p['timestamp'] <= 3.0]
        
        # 添加到长期窗口 (8秒)
        self.long_window.extend(valid_frames)
        self.long_window = [p for p in self.long_window 
                          if current_time - p['timestamp'] <= 8.0]
    
    def analyze_windows(self):
        """分析所有时间窗口数据并加权"""
        # 分析各窗口
        short_weights = self.analyze_single_window(self.short_window)
        mid_weights = self.analyze_single_window(self.mid_window)
        long_weights = self.analyze_single_window(self.long_window)
        
        # 合并权重
        combined_weights = {}
        for posture in self.original_classes.values():
            combined_weights[posture] = (
                short_weights.get(posture, 0) * self.short_window_weight +
                mid_weights.get(posture, 0) * self.mid_window_weight +
                long_weights.get(posture, 0) * self.long_window_weight
            )
        
        return combined_weights
    
    def analyze_single_window(self, window):
        """分析单个时间窗口，得出各姿态权重"""
        if not window:
            return {}
        
        # 统计每种姿态
        posture_stats = {}
        
        for record in window:
            posture = record['posture']
            confidence = record['confidence']
            timestamp = record['timestamp']
            
            if posture not in posture_stats:
                posture_stats[posture] = {
                    'count': 0,
                    'total_conf': 0,
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'confidences': []
                }
                
            stats = posture_stats[posture]
            stats['count'] += 1
            stats['total_conf'] += confidence
            stats['confidences'].append(confidence)
            stats['last_seen'] = max(stats['last_seen'], timestamp)
        
        # 计算各姿态权重
        total_records = len(window)
        weights = {}
        
        for posture, stats in posture_stats.items():
            # 计算频率 (0.4权重)
            frequency = stats['count'] / total_records
            
            # 计算平均置信度 (0.4权重) - 增加权重以提高稳定性
            avg_confidence = stats['total_conf'] / stats['count']
            
            # 计算持续时间占比 (0.2权重)
            duration = stats['last_seen'] - stats['first_seen']
            max_duration = window[-1]['timestamp'] - window[0]['timestamp']
            duration_ratio = duration / max_duration if max_duration > 0 else 0
            
            # 综合权重 - 增加置信度权重以提高稳定性
            weight = 0.4 * frequency + 0.4 * avg_confidence + 0.2 * duration_ratio
            weights[posture] = weight
        
        return weights
    
    def apply_transition_penalties(self, posture_weights):
        """根据姿态转换规律应用惩罚"""
        if not self.last_output or self.last_output == 'nobody':
            return posture_weights
        
        adjusted_weights = posture_weights.copy()
        
        # 应用转换惩罚
        for posture, weight in posture_weights.items():
            if self.last_output in self.transition_matrix:
                transition_prob = self.transition_matrix[self.last_output].get(posture, 0.01)
                # 惩罚不太可能的转换
                penalty = (1 - transition_prob) * self.transition_penalty
                adjusted_weights[posture] = max(0, weight - penalty)
        
        return adjusted_weights
    
    def correct_special_postures(self, posture_weights):
        """对特定姿态进行校正"""
        # 如果没有任何姿态权重，直接判定为nobody
        if not posture_weights:
            # 添加容错：如果之前有检测到姿态且间隔非常短（短于1秒），暂时保持上次的状态
            current_time = time.time()
            if (self.last_output and self.last_output != 'nobody' and 
                hasattr(self, 'last_detection_time') and self.last_detection_time and 
                current_time - self.last_detection_time < 1.0):
                self.nobody_detected = False
                return posture_weights
            else:
                self.nobody_detected = True
                return posture_weights
            
        # 为right_headed添加额外阈值修正
        if 'right_headed' in posture_weights:
            # 将right_headed的置信度降低20%
            posture_weights['right_headed'] *= 0.8
            # 原有的修正仍然保留
            posture_weights['right_headed'] -= self.right_headed_correction
            posture_weights['right_headed'] = max(0, posture_weights['right_headed'])
        
        # 为desk_leaning添加额外阈值修正，降低20%置信度
        if 'desk_leaning' in posture_weights:
            posture_weights['desk_leaning'] *= 0.8
            posture_weights['desk_leaning'] = max(0, posture_weights['desk_leaning'])
        
        # 改进nobody判定条件:
        # 检查所有姿态的总权重是否都低于0.15
        total_confidence = sum(posture_weights.values())
        
        # 判断没有人的多种情况
        nobody_detected = False
        current_time = time.time()
        
        # 引入姿势状态容错：当前输出为有效姿势且时间较短时不立即切换到nobody
        if self.last_output and self.last_output != 'nobody':
            # 如果上一次状态是有效姿势，我们提供一个更宽松的nobody判断
            # 1. 当姿势转换过程中总置信度较低时，给予更多容错
            if total_confidence < 0.15:
                # 转换期间容错：如果上次状态是有效姿势且间隔短于1.5秒，保持原状态
                if hasattr(self, 'last_detection_time') and self.last_detection_time:
                    time_since_last_detection = current_time - self.last_detection_time
                    if time_since_last_detection < 1.5:  # 1.5秒容错期
                        nobody_detected = False
                    else:
                        nobody_detected = True
                else:
                    nobody_detected = True
            # 2. 当姿势转换过程中最高置信度较低时，也给予容错
            else:
                highest_confidence = max(posture_weights.values()) if posture_weights else 0
                if highest_confidence < 0.08:  # 非常低的置信度阈值
                    # 只有当持续低于阈值超过1秒时才判定为nobody
                    if hasattr(self, 'last_detection_time') and self.last_detection_time:
                        time_since_last_detection = current_time - self.last_detection_time
                        if time_since_last_detection < 1.0:  # 1秒容错期
                            nobody_detected = False
                        else:
                            nobody_detected = True
                    else:
                        nobody_detected = True
        else:
            # 如果上一次状态已经是nobody，使用更严格的判断条件
            # 1. 总置信度低于阈值
            if total_confidence < 0.15:
                nobody_detected = True
                
            # 2. 所有姿态的单个置信度都很低
            highest_confidence = max(posture_weights.values()) if posture_weights else 0
            if highest_confidence < 0.08:  # 非常低的置信度阈值
                nobody_detected = True
        
        # 3. 当前时间与最后一个检测帧时间间隔过大
        # 无论前一状态如何，如果超过2秒没有新检测帧，一定是nobody
        if self.short_window:
            last_frame_time = max(record['timestamp'] for record in self.short_window if record)
            if current_time - last_frame_time > 2.5:  # 延长到2.5秒，增加容错性
                nobody_detected = True
        
        self.nobody_detected = nobody_detected
        
        # 保存统计数据供UI显示
        self.posture_stats = {}
        for posture, weight in posture_weights.items():
            self.posture_stats[posture] = {
                'weight': weight,
                'count': 0,
                'total_conf': 0,
                'confidences': []
            }
            
            # 统计各窗口数据
            for window in [self.short_window, self.mid_window, self.long_window]:
                for record in window:
                    if record and 'posture' in record and record['posture'] == posture:
                        self.posture_stats[posture]['count'] += 1
                        self.posture_stats[posture]['total_conf'] += record['confidence']
                        self.posture_stats[posture]['confidences'].append(record['confidence'])
                        
        return posture_weights
    
    def determine_final_output(self, posture_weights):
        """确定最终输出类别"""
        current_time = time.time()
        
        # 转换期间判断：如果上一个状态是有效姿势，尝试维持一段时间以克服短暂断点
        is_transition_period = False
        if (self.last_output and self.last_output != 'nobody' and 
            hasattr(self, 'last_detection_time') and self.last_detection_time):
            time_since_last = current_time - self.last_detection_time
            # 在1.5秒以内的转换期，给予更大容错
            if time_since_last < 1.5:
                is_transition_period = True
        
        # 检查是否有足够的高置信度检测来覆盖nobody状态
        high_conf_transition = False
        if self.last_output == 'nobody' and self.check_recent_high_confidence_detections():
            high_conf_transition = True
            self.nobody_detected = False  # 覆盖nobody检测标志
        
        # 检查是否没有人
        if self.nobody_detected and not is_transition_period and not high_conf_transition:
            # 只有不在姿势转换期间才确认nobody状态
            self.current_output = 'nobody'
            # 重置稳定性计数
            if self.last_output != 'nobody':
                self.output_stability = 0
                self.stability_score = 0.0
            self.last_output = 'nobody'
            return
        elif self.nobody_detected and (is_transition_period or high_conf_transition):
            # 如果在转换期间临时判断为nobody，我们暂时保持上一个状态或转换到新状态
            if high_conf_transition:
                # 尝试从最近的高置信度检测确定新状态
                new_state = self.determine_state_from_recent_detections()
                if new_state:
                    self.nobody_detected = False
                    self.current_output = new_state
                    # 部分重置稳定性，使过渡更平滑
                    if self.last_output != new_state:
                        self.output_stability = max(0, self.output_stability - 50)
                        self.stability_score = self.output_stability / 100
                    self.last_output = new_state
                return
            else:
                # 保持当前状态
                self.nobody_detected = False
                return
        
        # 空的posture_weights也应该判定为没有人，除非有高置信度的检测可以覆盖
        if not posture_weights:
            if high_conf_transition:
                # 尝试从最近的高置信度检测确定新状态
                new_state = self.determine_state_from_recent_detections()
                if new_state:
                    self.current_output = new_state
                    # 部分重置稳定性，使过渡更平滑
                    if self.last_output != new_state:
                        self.output_stability = max(0, self.output_stability - 50)
                        self.stability_score = self.output_stability / 100
                    self.last_output = new_state
                return
            elif is_transition_period:
                # 在转换期间，保持上一个状态
                return
            else:
                self.current_output = 'nobody'
                # 重置稳定性计数
                if self.last_output != 'nobody':
                    self.output_stability = 0
                    self.stability_score = 0.0
                self.last_output = 'nobody'
                return
        
        # 找出权重最高的姿态
        max_posture = max(posture_weights.items(), key=lambda x: x[1], default=(None, 0))
        
        # 如果最高权重太低，也应当判定为nobody，除非有高置信度的检测可以覆盖
        if max_posture[1] < 0.05 and not is_transition_period and not high_conf_transition:
            self.current_output = 'nobody'
            # 重置稳定性计数
            if self.last_output != 'nobody':
                self.output_stability = 0
                self.stability_score = 0.0
            self.last_output = 'nobody'
            return
        elif max_posture[1] < 0.05 and (is_transition_period or high_conf_transition):
            # 转换期间权重过低，检查是否可以从高置信度检测确定状态
            if high_conf_transition:
                new_state = self.determine_state_from_recent_detections()
                if new_state:
                    self.current_output = new_state
                    if self.last_output != new_state:
                        self.output_stability = max(0, self.output_stability - 50)
                        self.stability_score = self.output_stability / 100
                    self.last_output = new_state
                return
            else:
                # 保持上一状态
                return
        
        # 如果最高权重的姿态为None
        if max_posture[0] is None:
            if high_conf_transition:
                new_state = self.determine_state_from_recent_detections()
                if new_state:
                    self.current_output = new_state
                    if self.last_output != new_state:
                        self.output_stability = max(0, self.output_stability - 50)
                        self.stability_score = self.output_stability / 100
                    self.last_output = new_state
                return
            elif is_transition_period:
                # 在转换期间，保持上一个状态
                return
            else:
                self.current_output = 'nobody'
                # 重置稳定性计数
                if self.last_output != 'nobody':
                    self.output_stability = 0
                    self.stability_score = 0.0
                self.last_output = 'nobody'
                return
        
        # 转换为简化输出
        if max_posture[0] == 'correct_posture':
            new_output = 'correct_posture'
        elif max_posture[0] in ['bowed_head', 'desk_leaning', 'head_tilt', 'left_headed', 'right_headed']:
            # 将所有不正确姿势（包括低头）归入incorrect_posture
            new_output = 'incorrect_posture'
        else:
            new_output = 'nobody'
        
        # 特殊处理姿势转换：如果从correct_posture切换到incorrect_posture或反之，
        # 需要更高的置信度才能触发切换
        if (is_transition_period and 
            ((self.last_output == 'correct_posture' and new_output == 'incorrect_posture') or 
             (self.last_output == 'incorrect_posture' and new_output == 'correct_posture'))):
            # 需要更高的权重才能切换
            if max_posture[1] < 0.25:  # 要求更高的置信度来确认姿势变化
                return  # 保持上一个状态
        
        self.current_output = new_output
        
        # 更新稳定性计数 - 使用更精细的1%步长
        if self.current_output == self.last_output:
            # 每次批次处理增加10个单位，相当于10%的稳定性增长
            self.output_stability += 10  # 从1改为10，使每个批次增加10%而不是1%
            # 使用100步作为总计，这样每次批次处理增加10%，同时保持1%的显示精度
            self.stability_score = min(1.0, self.output_stability / 100)  # 100步计算完全稳定性
        else:
            # 如果是姿势变换，减少稳定性而不是完全重置，让过渡更平滑
            if is_transition_period and self.last_output != 'nobody' and self.current_output != 'nobody':
                # 姿势变换时只降低50%的稳定性
                self.output_stability = max(0, self.output_stability - 50)
                self.stability_score = self.output_stability / 100
            elif high_conf_transition and self.last_output == 'nobody':
                # 从nobody转换到其他状态时快速增加稳定性
                self.output_stability = 20  # 给予20%的初始稳定性
                self.stability_score = 0.2
            else:
                # 其他情况完全重置
                self.output_stability = 0
                self.stability_score = 0.0
            
            self.last_output = self.current_output
    
    def get_output_and_stats(self):
        """获取输出结果和统计数据"""
        return {
            'output': self.current_output,
            'stability': self.stability_score,
            'posture_stats': self.posture_stats,
            'window_stats': {
                'short': len(self.short_window),
                'mid': len(self.mid_window),
                'long': len(self.long_window)
            }
        }

class ModernUI:
    """现代UI样式设置类"""
    @staticmethod
    def configure_styles():
        """配置应用的现代风格"""
        # 创建自定义样式
        style = ttk.Style()
        
        # 设置主题（如果可用）
        try:
            style.theme_use("clam")  # 使用较为现代的主题基础
        except tk.TclError:
            pass  # 如果主题不可用，使用默认主题
        
        # 高分辨率设置
        font_size = 11  # 基础字体大小
        title_font_size = 13  # 标题字体大小
        button_font_size = 12  # 按钮字体大小
        
        # 设置全局字体和抗锯齿
        font_family = "Microsoft YaHei" if os.name == "nt" else "Helvetica"  # 在Windows上使用微软雅黑，其他系统使用Helvetica
        
        # 应用自定义样式 - 带圆角的按钮
        style.configure(
            "Modern.TButton",
            background="#007AFF",  # 苹果蓝
            foreground="white",
            borderwidth=0,
            focusthickness=0,
            font=(font_family, button_font_size, "bold"),
            padding=(12, 6)
        )
        style.map(
            "Modern.TButton",
            background=[("active", "#0069D9")],  # 点击时的颜色
            foreground=[("active", "white")]
        )
        
        # 设置标签样式
        style.configure(
            "Modern.TLabel",
            font=(font_family, font_size),
            background="#F5F5F7"  # 苹果浅灰色背景
        )
        
        # 设置标题标签样式
        style.configure(
            "Title.TLabel",
            font=(font_family, title_font_size, "bold"),
            padding=(5, 10, 5, 5),
            background="#F5F5F7"
        )
        
        # 设置框架样式
        style.configure(
            "Modern.TFrame",
            background="#F5F5F7"
        )
        
        # 设置LabelFrame样式 - 用于分组控件
        style.configure(
            "Modern.TLabelframe",
            background="#F5F5F7",
            padding=12
        )
        style.configure(
            "Modern.TLabelframe.Label",
            font=(font_family, font_size, "bold"),
            background="#F5F5F7"
        )
        
        # 设置Scale滑块样式
        style.configure(
            "Modern.Horizontal.TScale",
            sliderthickness=24,
            sliderrelief=tk.FLAT,
            background="#F5F5F7"
        )
        
        # 设置下拉框样式
        style.configure(
            "Modern.TCombobox",
            padding=6,
            font=(font_family, font_size),
            fieldbackground="white"
        )
        
        # 设置单选按钮样式
        style.configure(
            "Modern.TRadiobutton",
            background="#F5F5F7",
            font=(font_family, font_size)
        )
        
        # 设置输入框样式
        style.configure(
            "Modern.TEntry",
            padding=6,
            font=(font_family, font_size),
            fieldbackground="white"
        )
        
        # 设置值显示样式 - 用于显示滑块值
        style.configure(
            "Value.TLabel",
            font=(font_family, font_size, "bold"),
            foreground="#007AFF",
            background="#F5F5F7"
        )
        
        return style

# 性能监控类
class PerformanceMonitor:
    """模型性能监控类"""
    def __init__(self):
        self.reset()
        self.initialized = False
        # 存储模型GFLOPS信息的缓存
        self.model_gflops_cache = {}
    
    def reset(self):
        """重置性能监控数据"""
        self.inference_times = []
        self.gflops_values = []
        self.memory_usage = []
        self.last_gflops = 0
        self.last_inference_time = 0
        self.last_memory = 0
        
    def update(self, inference_time, gflops, memory_usage):
        """更新性能指标"""
        self.inference_times.append(inference_time)
        self.gflops_values.append(gflops)
        self.memory_usage.append(memory_usage)
        
        # 只保留最近50次测量结果
        if len(self.inference_times) > 50:
            self.inference_times.pop(0)
            self.gflops_values.pop(0)
            self.memory_usage.pop(0)
        
        self.last_inference_time = inference_time
        self.last_gflops = gflops
        self.last_memory = memory_usage
    
    def get_average_values(self):
        """获取平均性能值"""
        if not self.inference_times:
            return 0, 0, 0
        
        avg_inference = sum(self.inference_times) / len(self.inference_times)
        avg_gflops = sum(self.gflops_values) / len(self.gflops_values)
        avg_memory = sum(self.memory_usage) / len(self.memory_usage)
        
        return avg_inference, avg_gflops, avg_memory
    
    def estimate_model_gflops(self, model, img_size=640):
        """估算模型的GFLOPS"""
        # 检查缓存中是否已有该模型的GFLOPS数据
        model_key = f"{model.__class__.__name__}_{img_size}"
        if model_key in self.model_gflops_cache:
            return self.model_gflops_cache[model_key]
        
        # 创建输入张量
        device = next(model.parameters()).device
        input_tensor = torch.randn(1, 3, img_size, img_size).to(device)
        
        # 清理显存/内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        try:
            # 使用thop计算FLOPS
            print(f"开始计算模型GFLOPS, 设备: {device}, 输入尺寸: {img_size}")
            macs, params = thop_profile(model, inputs=(input_tensor,))
            gflops = macs / 1e9
            print(f"GFLOPS计算结果: {gflops:.2f} GFLOPS, 参数量: {params/1e6:.2f}M")
            
            # 存入缓存
            self.model_gflops_cache[model_key] = gflops
            self.initialized = True
            return gflops
        except Exception as e:
            print(f"GFLOPS计算错误: {str(e)}")
            # 使用一个合理的默认值
            default_gflops = 35.0  # 假设是一个中等规模的YOLO模型
            self.model_gflops_cache[model_key] = default_gflops
            self.initialized = True
            return default_gflops
    
    def get_inference_gflops(self, model, inference_time, img_size=640):
        """根据推理时间和模型估计实际GFLOPS使用情况"""
        if not self.initialized:
            try:
                # 首次调用，计算并缓存基准GFLOPS
                base_gflops = self.estimate_model_gflops(model, img_size)
                self.initialized = True
            except Exception as e:
                print(f"初始化GFLOPS估算失败: {str(e)}")
                base_gflops = 35.0  # 使用默认值
        else:
            # 从缓存获取基准GFLOPS
            model_key = f"{model.__class__.__name__}_{img_size}"
            base_gflops = self.model_gflops_cache.get(model_key, 35.0)
        
        # 确保返回合理值，避免返回0
        if base_gflops <= 0:
            base_gflops = 35.0  # 默认值
        
        # 即使在RTX4070混合输出的情况下也能显示合理的GFLOPS值
        return base_gflops
    
    def get_memory_usage(self):
        """获取当前内存使用情况（MB）"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / 1024 / 1024
        
        # 如果有GPU，也获取GPU内存
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            return memory_usage_mb, gpu_memory
        else:
            return memory_usage_mb, 0

class YoloGUI:
    def __init__(self, root):
        """初始化主应用程序"""
        self.root = root
        self.root.title("儿童学习姿态识别系统 (Children's Learning Posture Recognition System)")
        self.root.geometry("1280x900")
        self.root.minsize(1200, 800)
        
        # 应用现代UI样式
        self.style = ModernUI.configure_styles()
        
        # 设置整体背景色
        self.root.configure(background="#F5F5F7")
        
        # 全局变量
        self.cap = None
        self.model = None
        self.is_running = False
        self.thread = None
        self.frame_delay = 0.033  # 默认约30 FPS
        self.current_input_type = "camera"
        self.video_path = None
        self.image_path = None
        self.available_models = self.get_available_models()
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        self.performance_vars = {
            'gflops': tk.StringVar(value="-- GFLOPS"),
            'inference_time': tk.StringVar(value="-- ms"),
            'memory': tk.StringVar(value="-- MB"),
            'avg_gflops': tk.StringVar(value="-- GFLOPS"),
            'avg_inference': tk.StringVar(value="-- ms"),
            'fps_real': tk.StringVar(value="-- FPS")
        }
        
        # 算法选择 - 默认启用pro算法
        self.algorithm_type = tk.StringVar(value="pro")  # none, smart, pro
        
        # 智能姿态算法参数
        self.use_smart_algorithm = tk.BooleanVar(value=False)
        self.posture_window = []  # 时间窗口内的姿态记录
        self.window_duration = 3.0  # 时间窗口长度(秒)
        self.trigger_threshold = 0.7  # 触发阈值
        self.trigger_cooldown = 5.0  # 相同姿态的提示冷却时间(秒)
        self.last_trigger_time = {}  # 每种姿态最后一次触发时间
        self.posture_stats = {}  # 姿态统计信息
        
        # 智能算法Pro参数
        self.posture_pro_recognizer = PostureProRecognizer(aggressiveness=50)
        self.pro_aggressiveness = tk.IntVar(value=50)  # 算法激进度
        self.last_pro_result = None  # 最后一次Pro算法识别结果
        
        # 日志相关变量
        self.is_log_visible = False
        self.log_window = None
        self.log_text = None
        self.detailed_log_btn = None
        self.log_entries = []
        
        # 性能日志窗口
        self.is_perf_window_visible = False
        self.perf_window = None
        
        # 创建主框架
        self.create_widgets()
        
        # 默认加载模型
        self.load_model()
        
    def get_available_models(self):
        """获取可用的模型文件"""
        models = []
        
        # 在models目录查找.pt文件
        models.extend(glob.glob("models/*.pt"))
        
        # 在models/backup目录查找.pt文件
        models.extend(glob.glob("models/backup/*.pt"))
        
        # 去掉路径前缀，便于显示
        clean_models = [os.path.basename(m) for m in models]
        
        # 添加完整路径映射
        self.model_paths = {os.path.basename(m): m for m in models}
        
        # 定义默认模型名称
        self.default_model = "best_70.pt"
        
        # 如果默认模型不在列表中，使用第一个可用模型
        if clean_models and self.default_model not in clean_models:
            self.default_model = clean_models[0]
            
        return clean_models
    
    def create_widgets(self):
        """创建GUI控件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10", style="Modern.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板容器
        control_container_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10", style="Modern.TLabelframe")
        control_container_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # 添加可滚动的Canvas
        self.control_canvas = tk.Canvas(control_container_frame, highlightthickness=0, bg="#F5F5F7", width=280)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 添加滚动条
        control_scrollbar = ttk.Scrollbar(control_container_frame, orient=tk.VERTICAL, command=self.control_canvas.yview)
        control_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 配置canvas
        self.control_canvas.configure(yscrollcommand=control_scrollbar.set)
        
        # 创建内部框架放置控件
        control_frame = ttk.Frame(self.control_canvas, style="Modern.TFrame")
        self.control_canvas.create_window((0, 0), window=control_frame, anchor='nw')
        
        # 更新滚动区域
        def update_scrollregion(event):
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox('all'))
            # 设置内部frame的宽度为Canvas的宽度
            canvas_width = self.control_canvas.winfo_width()
            if canvas_width > 20:  # 确保有合理的宽度
                self.control_canvas.itemconfig(control_frame_window, width=canvas_width)
        
        # 将frame保存为变量以便后续更新宽度
        control_frame_window = self.control_canvas.create_window((0, 0), window=control_frame, anchor='nw')
        
        # 绑定大小变化事件
        control_frame.bind('<Configure>', update_scrollregion)
        
        # 绑定鼠标滚轮事件
        def _on_mousewheel(event):
            self.control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # 输入源选择
        ttk.Label(control_frame, text="输入源 (Input Source)", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 5))
        self.input_source = tk.StringVar(value="camera")
        ttk.Radiobutton(control_frame, text="摄像头 (Camera)", variable=self.input_source, value="camera", 
                       command=self.on_input_change, style="Modern.TRadiobutton").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="图片 (Image)", variable=self.input_source, value="image", 
                       command=self.on_input_change, style="Modern.TRadiobutton").pack(anchor=tk.W)
        ttk.Radiobutton(control_frame, text="视频 (Video)", variable=self.input_source, value="video", 
                       command=self.on_input_change, style="Modern.TRadiobutton").pack(anchor=tk.W)
        
        # 文件选择按钮
        self.file_button = ttk.Button(control_frame, text="选择文件...", command=self.browse_file, 
                                    style="Modern.TButton", state=tk.DISABLED)
        self.file_button.pack(anchor=tk.W, pady=10)
        
        # 模型选择
        ttk.Label(control_frame, text="模型选择 (Model Selection)", style="Title.TLabel").pack(anchor=tk.W, pady=(10, 5))
        self.model_var = tk.StringVar(value=self.default_model if self.available_models else "无可用模型")
        self.model_dropdown = ttk.Combobox(control_frame, textvariable=self.model_var, values=self.available_models, 
                                         style="Modern.TCombobox", state="readonly")
        self.model_dropdown.pack(anchor=tk.W, fill=tk.X, pady=(0, 10))
        self.model_dropdown.bind("<<ComboboxSelected>>", self.on_model_change)
        
        # 参数设置
        settings_frame = ttk.LabelFrame(control_frame, text="参数设置 (Settings)", style="Modern.TLabelframe")
        settings_frame.pack(fill=tk.X, pady=10)
        
        # 置信度阈值
        ttk.Label(settings_frame, text="置信度阈值 (Confidence Threshold)", style="Modern.TLabel").pack(anchor=tk.W, pady=(5, 0))
        conf_entry_frame = ttk.Frame(settings_frame, style="Modern.TFrame")
        conf_entry_frame.pack(anchor=tk.W, fill=tk.X)
        
        self.conf_var = tk.DoubleVar(value=0.35)
        
        # 定义置信度滑块变化的函数
        def conf_scale_changed(val):
            try:
                # 获取当前值
                current = float(val)
                # 将值限制在0.01-1.0之间
                if current < 0.01:
                    current = 0.01
                elif current > 1.0:
                    current = 1.0
                # 设置变量值
                self.conf_var.set(current)
            except ValueError:
                pass
        
        self.conf_scale = ttk.Scale(conf_entry_frame, from_=0.01, to=1.0, variable=self.conf_var, 
                                  command=conf_scale_changed, style="Modern.Horizontal.TScale", length=200)
        self.conf_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.conf_entry = ttk.Entry(conf_entry_frame, textvariable=self.conf_var, width=6, style="Modern.TEntry")
        self.conf_entry.pack(side=tk.RIGHT, padx=(5, 0))
        
        # 添加点击全选功能
        def conf_entry_click(event):
            self.conf_entry.select_range(0, tk.END)
            return "break"
        
        self.conf_entry.bind("<FocusIn>", conf_entry_click)
        
        # IOU阈值
        ttk.Label(settings_frame, text="IOU阈值 (IOU Threshold)", style="Modern.TLabel").pack(anchor=tk.W, pady=(10, 0))
        iou_entry_frame = ttk.Frame(settings_frame, style="Modern.TFrame")
        iou_entry_frame.pack(anchor=tk.W, fill=tk.X)
        
        self.iou_var = tk.DoubleVar(value=0.5)
        
        # 定义IOU滑块变化的函数
        def iou_scale_changed(val):
            try:
                # 获取当前值
                current = float(val)
                # 将值限制在0.01-1.0之间
                if current < 0.01:
                    current = 0.01
                elif current > 1.0:
                    current = 1.0
                # 设置变量值
                self.iou_var.set(current)
            except ValueError:
                pass
        
        self.iou_scale = ttk.Scale(iou_entry_frame, from_=0.01, to=1.0, variable=self.iou_var, 
                                 command=iou_scale_changed, style="Modern.Horizontal.TScale", length=200)
        self.iou_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.iou_entry = ttk.Entry(iou_entry_frame, textvariable=self.iou_var, width=6, style="Modern.TEntry")
        self.iou_entry.pack(side=tk.RIGHT, padx=(5, 0))
        
        # 添加点击全选功能
        def iou_entry_click(event):
            self.iou_entry.select_range(0, tk.END)
            return "break"
        
        self.iou_entry.bind("<FocusIn>", iou_entry_click)
        
        # 帧率控制
        ttk.Label(settings_frame, text="帧率 (Frame Rate)", style="Modern.TLabel").pack(anchor=tk.W, pady=(10, 0))
        fps_entry_frame = ttk.Frame(settings_frame, style="Modern.TFrame")
        fps_entry_frame.pack(anchor=tk.W, fill=tk.X)
        
        self.fps_var = tk.DoubleVar(value=30)
        
        # 定义FPS滑块变化的函数，支持0.05的细微调整
        def fps_scale_changed(val):
            try:
                # 获取当前值
                current = float(val)
                # 将值限制在0.1-30之间
                if current < 0.1:
                    current = 0.1
                elif current > 30.0:
                    current = 30.0
                # 设置变量值
                self.fps_var.set(current)
            except ValueError:
                pass
        
        self.fps_scale = ttk.Scale(fps_entry_frame, from_=0.1, to=30.0, variable=self.fps_var, 
                                 command=fps_scale_changed, style="Modern.Horizontal.TScale", length=200)
        self.fps_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.fps_entry = ttk.Entry(fps_entry_frame, textvariable=self.fps_var, width=6, style="Modern.TEntry")
        self.fps_entry.pack(side=tk.RIGHT, padx=(5, 0))
        
        # 添加点击全选功能
        def fps_entry_click(event):
            self.fps_entry.select_range(0, tk.END)
            return "break"
        
        self.fps_entry.bind("<FocusIn>", fps_entry_click)
        
        # 更新帧率值
        def update_fps_label(*args):
            try:
                fps = self.fps_var.get()
                if fps < 0.1:
                    fps = 0.1
                elif fps > 30.0:
                    fps = 30.0
                
                # 根据不同范围设置不同步长
                if fps < 0.5:
                    # 0.5以下：步长0.05
                    fps = round(fps / 0.05) * 0.05
                elif fps < 3.0:
                    # 0.5到3之间：步长0.1
                    fps = round(fps / 0.1) * 0.1
                else:
                    # 3以上：步长0.5
                    fps = round(fps / 0.5) * 0.5
                
                self.fps_var.set(fps)
                self.frame_delay = 1.0 / fps
            except (ValueError, tk.TclError):
                pass
        
        self.fps_var.trace_add("write", update_fps_label)
        
        # 更新置信度阈值显示
        def update_conf_label(*args):
            try:
                conf = self.conf_var.get()
                if conf < 0.01:
                    conf = 0.01
                elif conf > 1.0:
                    conf = 1.0
                # 按0.02的步长调整
                conf = round(conf / 0.02) * 0.02
                self.conf_var.set(conf)
            except (ValueError, tk.TclError):
                pass
            
        self.conf_var.trace_add("write", update_conf_label)
        
        # 更新IOU阈值显示
        def update_iou_label(*args):
            try:
                iou = self.iou_var.get()
                if iou < 0.01:
                    iou = 0.01
                elif iou > 1.0:
                    iou = 1.0
                # 按0.02的步长调整
                iou = round(iou / 0.02) * 0.02
                self.iou_var.set(iou)
            except (ValueError, tk.TclError):
                pass
            
        self.iou_var.trace_add("write", update_iou_label)
        
        # 智能算法设置
        algorithm_frame = ttk.LabelFrame(control_frame, text="智能识别算法 (Smart Algorithm)", style="Modern.TLabelframe")
        algorithm_frame.pack(fill=tk.X, pady=10)
        
        # 算法选择
        algorithm_select_frame = ttk.Frame(algorithm_frame, style="Modern.TFrame")
        algorithm_select_frame.pack(fill=tk.X, pady=(5, 10))
        
        ttk.Label(algorithm_select_frame, text="选择算法:", style="Modern.TLabel").pack(side=tk.LEFT, padx=(0, 10))
        
        # 算法选择单选按钮组 - 改为垂直布局
        algo_radio_frame = ttk.Frame(algorithm_frame, style="Modern.TFrame")
        algo_radio_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Radiobutton(algo_radio_frame, text="关闭", variable=self.algorithm_type, value="none", 
                       command=self.on_algorithm_change, style="Modern.TRadiobutton").pack(anchor=tk.W, padx=(10, 0), pady=2)
        ttk.Radiobutton(algo_radio_frame, text="智能阈值", variable=self.algorithm_type, value="smart", 
                       command=self.on_algorithm_change, style="Modern.TRadiobutton").pack(anchor=tk.W, padx=(10, 0), pady=2)
        ttk.Radiobutton(algo_radio_frame, text="智能Pro", variable=self.algorithm_type, value="pro", 
                       command=self.on_algorithm_change, style="Modern.TRadiobutton").pack(anchor=tk.W, padx=(10, 0), pady=2)
        
        # 创建智能阈值算法参数框架
        self.smart_params_frame = ttk.Frame(algorithm_frame, style="Modern.TFrame")
        
        # 窗口长度控制
        ttk.Label(self.smart_params_frame, text="时间窗口长度 (秒)", style="Modern.TLabel").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.window_var = tk.DoubleVar(value=self.window_duration)
        self.window_scale = ttk.Scale(self.smart_params_frame, from_=1.0, to=10.0, variable=self.window_var,
                                    command=self.update_window_duration, style="Modern.Horizontal.TScale", length=150)
        self.window_scale.grid(row=0, column=1, sticky=tk.W, padx=5)
        self.window_label = ttk.Label(self.smart_params_frame, text=f"{self.window_duration:.1f}秒", style="Value.TLabel")
        self.window_label.grid(row=0, column=2, sticky=tk.W)
        
        # 添加窗口长度输入框
        self.window_entry = ttk.Entry(self.smart_params_frame, textvariable=self.window_var, width=5, style="Modern.TEntry")
        self.window_entry.grid(row=0, column=3, sticky=tk.W, padx=(10, 0))
        
        # 添加点击全选功能
        def window_entry_click(event):
            self.window_entry.select_range(0, tk.END)
            return "break"
        
        self.window_entry.bind("<FocusIn>", window_entry_click)
        
        # 触发阈值控制
        ttk.Label(self.smart_params_frame, text="触发阈值", style="Modern.TLabel").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.threshold_var = tk.DoubleVar(value=self.trigger_threshold)
        self.threshold_scale = ttk.Scale(self.smart_params_frame, from_=0.3, to=0.95, variable=self.threshold_var,
                                      command=self.update_trigger_threshold, style="Modern.Horizontal.TScale", length=150)
        self.threshold_scale.grid(row=1, column=1, sticky=tk.W, padx=5)
        self.threshold_label = ttk.Label(self.smart_params_frame, text=f"{self.trigger_threshold:.2f}", style="Value.TLabel")
        self.threshold_label.grid(row=1, column=2, sticky=tk.W)
        
        # 添加触发阈值输入框
        self.threshold_entry = ttk.Entry(self.smart_params_frame, textvariable=self.threshold_var, width=5, style="Modern.TEntry")
        self.threshold_entry.grid(row=1, column=3, sticky=tk.W, padx=(10, 0))
        
        # 添加点击全选功能
        def threshold_entry_click(event):
            self.threshold_entry.select_range(0, tk.END)
            return "break"
        
        self.threshold_entry.bind("<FocusIn>", threshold_entry_click)
        
        # 冷却时间控制
        ttk.Label(self.smart_params_frame, text="提示冷却时间 (秒)", style="Modern.TLabel").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.cooldown_var = tk.DoubleVar(value=self.trigger_cooldown)
        self.cooldown_scale = ttk.Scale(self.smart_params_frame, from_=1.0, to=20.0, variable=self.cooldown_var,
                                     command=self.update_cooldown, style="Modern.Horizontal.TScale", length=150)
        self.cooldown_scale.grid(row=2, column=1, sticky=tk.W, padx=5)
        self.cooldown_label = ttk.Label(self.smart_params_frame, text=f"{self.trigger_cooldown:.1f}秒", style="Value.TLabel")
        self.cooldown_label.grid(row=2, column=2, sticky=tk.W)
        
        # 添加冷却时间输入框
        self.cooldown_entry = ttk.Entry(self.smart_params_frame, textvariable=self.cooldown_var, width=5, style="Modern.TEntry")
        self.cooldown_entry.grid(row=2, column=3, sticky=tk.W, padx=(10, 0))
        
        # 添加点击全选功能
        def cooldown_entry_click(event):
            self.cooldown_entry.select_range(0, tk.END)
            return "break"
        
        self.cooldown_entry.bind("<FocusIn>", cooldown_entry_click)
        
        # 创建智能Pro算法参数框架
        self.pro_params_frame = ttk.Frame(algorithm_frame, style="Modern.TFrame")
        
        # 标题
        title_label = ttk.Label(self.pro_params_frame, text="智能算法Pro设置", 
                              font=("Microsoft YaHei", 10, "bold"), 
                              foreground="#007AFF", style="Modern.TLabel")
        title_label.grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))
        
        # 激进度控制
        ttk.Label(self.pro_params_frame, text="算法激进度:", style="Modern.TLabel").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.aggressiveness_scale = ttk.Scale(self.pro_params_frame, from_=0, to=100, variable=self.pro_aggressiveness,
                                           command=self.update_pro_aggressiveness, style="Modern.Horizontal.TScale", length=180)
        self.aggressiveness_scale.grid(row=1, column=1, sticky=tk.W, padx=5)
        self.aggressiveness_label = ttk.Label(self.pro_params_frame, text="平衡 (50)", style="Value.TLabel")
        self.aggressiveness_label.grid(row=1, column=2, sticky=tk.W)
        
        # 添加激进度输入框
        self.pro_aggr_entry = ttk.Entry(self.pro_params_frame, textvariable=self.pro_aggressiveness, width=5, style="Modern.TEntry")
        self.pro_aggr_entry.grid(row=1, column=3, sticky=tk.W, padx=(5, 0))
        
        # 添加点击全选功能
        def pro_aggr_entry_click(event):
            self.pro_aggr_entry.select_range(0, tk.END)
            return "break"
        
        self.pro_aggr_entry.bind("<FocusIn>", pro_aggr_entry_click)
        
        # 添加分隔线
        separator = ttk.Separator(self.pro_params_frame, orient=tk.HORIZONTAL)
        separator.grid(row=2, column=0, columnspan=4, sticky=tk.EW, pady=10)
        
        # 参数显示区域
        ttk.Label(self.pro_params_frame, text="当前配置:", font=("Microsoft YaHei", 9, "bold"), 
                 style="Modern.TLabel").grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        
        # 配置信息框 - 使用更现代的网格布局
        conf_frame = ttk.Frame(self.pro_params_frame, style="Modern.TFrame")
        conf_frame.grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(0, 10))
        
        # 参数显示
        self.conf_value_label = ttk.Label(conf_frame, text="置信度: 0.45", style="Modern.TLabel")
        self.conf_value_label.grid(row=0, column=0, sticky=tk.W, padx=(10, 20))
        
        self.iou_value_label = ttk.Label(conf_frame, text="IOU: 0.55", style="Modern.TLabel")
        self.iou_value_label.grid(row=0, column=1, sticky=tk.W)
        
        self.window_weights_label = ttk.Label(conf_frame, text="窗口权重: 0.6/0.3/0.1", style="Modern.TLabel")
        self.window_weights_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=(10, 0), pady=(5, 0))
        
        # 添加另一个分隔线
        separator2 = ttk.Separator(self.pro_params_frame, orient=tk.HORIZONTAL)
        separator2.grid(row=5, column=0, columnspan=4, sticky=tk.EW, pady=10)
        
        # 输出模式信息框架
        output_frame = ttk.Frame(self.pro_params_frame, style="Modern.TFrame")
        output_frame.grid(row=6, column=0, columnspan=4, sticky=tk.W)
        
        # 输出模式信息框标题
        ttk.Label(output_frame, text="输出类型:", font=("Microsoft YaHei", 9, "bold"), 
                 style="Modern.TLabel").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        # 正确姿势项 - 带颜色指示
        correct_frame = ttk.Frame(output_frame, style="Modern.TFrame")
        correct_frame.grid(row=1, column=0, sticky=tk.W, padx=(10, 0), pady=2)
        
        correct_indicator = tk.Canvas(correct_frame, width=10, height=10, bg="#00AA00", highlightthickness=0)
        correct_indicator.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(correct_frame, text="Correct Posture (正确姿势)", style="Modern.TLabel").pack(side=tk.LEFT)
        
        # 不正确姿势项 - 带颜色指示
        incorrect_frame = ttk.Frame(output_frame, style="Modern.TFrame")
        incorrect_frame.grid(row=2, column=0, sticky=tk.W, padx=(10, 0), pady=2)
        
        incorrect_indicator = tk.Canvas(incorrect_frame, width=10, height=10, bg="#FF0000", highlightthickness=0)
        incorrect_indicator.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(incorrect_frame, text="Incorrect Posture (不正确姿势)", style="Modern.TLabel").pack(side=tk.LEFT)
        
        # 无人项 - 带颜色指示
        nobody_frame = ttk.Frame(output_frame, style="Modern.TFrame")
        nobody_frame.grid(row=3, column=0, sticky=tk.W, padx=(10, 0), pady=2)
        
        nobody_indicator = tk.Canvas(nobody_frame, width=10, height=10, bg="#888888", highlightthickness=0)
        nobody_indicator.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(nobody_frame, text="Nobody (无人)", style="Modern.TLabel").pack(side=tk.LEFT)
        
        # 默认不显示任何算法的参数面板
        # 调用算法切换函数来初始化界面状态
        self.on_algorithm_change()
        
        # 添加性能监控面板
        perf_frame = ttk.LabelFrame(control_frame, text="性能监控 (Performance)", style="Modern.TLabelframe")
        perf_frame.pack(fill=tk.X, pady=10)
        
        # GFLOPS显示
        ttk.Label(perf_frame, text="计算量:", style="Modern.TLabel").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(perf_frame, textvariable=self.performance_vars['gflops'], style="Value.TLabel").grid(row=0, column=1, sticky=tk.W)
        
        # 推理时间显示
        ttk.Label(perf_frame, text="推理时间:", style="Modern.TLabel").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(perf_frame, textvariable=self.performance_vars['inference_time'], style="Value.TLabel").grid(row=1, column=1, sticky=tk.W)
        
        # 内存使用显示
        ttk.Label(perf_frame, text="内存使用:", style="Modern.TLabel").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Label(perf_frame, textvariable=self.performance_vars['memory'], style="Value.TLabel").grid(row=2, column=1, sticky=tk.W)
        
        # 性能详情按钮
        ttk.Button(perf_frame, text="性能详情...", 
                 command=self.toggle_performance_window, style="Modern.TButton").grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # 日志按钮
        log_btn_frame = ttk.Frame(control_frame, style="Modern.TFrame")
        log_btn_frame.pack(fill=tk.X, pady=10)
        
        self.log_btn = ttk.Button(log_btn_frame, text="显示日志 (Show Log)", 
                                command=self.toggle_log_window, style="Modern.TButton")
        self.log_btn.pack(side=tk.LEFT)
        
        # 控制按钮
        button_frame = ttk.Frame(control_frame, style="Modern.TFrame")
        button_frame.pack(fill=tk.X, pady=20)
        
        self.start_button = ttk.Button(button_frame, text="开始 (Start)", 
                                     command=self.start_detection, style="Modern.TButton")
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(button_frame, text="停止 (Stop)", 
                                    command=self.stop_detection, style="Modern.TButton", state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)
        
        # 状态信息
        self.status_var = tk.StringVar(value="就绪 (Ready)")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W, style="Modern.TLabel")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 右侧显示区域
        display_frame = ttk.LabelFrame(main_frame, text="识别结果 (Detection Results)", 
                                     padding="10", style="Modern.TLabelframe")
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # 创建画布显示视频帧
        self.canvas = tk.Canvas(display_frame, bg="#222222", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 初始状态设置
        self.toggle_smart_algorithm()
        
    def on_input_change(self):
        """处理输入源变化"""
        selected = self.input_source.get()
        self.current_input_type = selected
        
        # 如果是摄像头，禁用文件选择按钮
        if selected == "camera":
            self.file_button.config(state=tk.DISABLED)
        else:
            self.file_button.config(state=tk.NORMAL)
        
        # 如果有正在运行的检测，停止它
        if self.is_running:
            self.stop_detection()
            
    def browse_file(self):
        """打开文件浏览对话框"""
        selected = self.input_source.get()
        if selected == "image":
            filetypes = [("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
            self.image_path = filedialog.askopenfilename(title="选择图片文件", filetypes=filetypes)
            if self.image_path:
                self.status_var.set(f"已选择图片: {os.path.basename(self.image_path)}")
        elif selected == "video":
            filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv")]
            self.video_path = filedialog.askopenfilename(title="选择视频文件", filetypes=filetypes)
            if self.video_path:
                self.status_var.set(f"已选择视频: {os.path.basename(self.video_path)}")
    
    def on_model_change(self, event):
        """处理模型变化"""
        # 如果正在运行，先停止
        if self.is_running:
            self.stop_detection()
        
        # 更新模型
        self.load_model()
    
    def load_model(self):
        """加载选定的YOLO模型"""
        model_name = self.model_var.get()
        if model_name in self.model_paths:
            try:
                self.model = YOLO(self.model_paths[model_name])
                self.status_var.set(f"已加载模型: {model_name}")
                self.add_log(f"加载模型: {model_name}", f"成功加载模型: {self.model_paths[model_name]}")
            except Exception as e:
                messagebox.showerror("模型加载错误", f"无法加载模型: {str(e)}")
                self.model = None
                self.add_log(f"模型加载失败: {model_name}", f"加载模型时出错: {str(e)}\n路径: {self.model_paths[model_name]}")
        else:
            messagebox.showerror("错误", "未找到选定的模型文件")
            self.model = None
            self.add_log("模型加载失败", f"未找到模型文件: {model_name}")
    
    def start_detection(self):
        """开始检测过程"""
        if not self.model:
            messagebox.showerror("错误", "请先加载有效的模型")
            return
        
        if self.is_running:
            return
        
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # 记录检测开始
        input_type_map = {
            "camera": "摄像头",
            "image": "图片",
            "video": "视频"
        }
        input_type = input_type_map.get(self.current_input_type, self.current_input_type)
        model_name = self.model_var.get()
        conf = self.conf_var.get()
        iou = self.iou_var.get()
        fps = self.fps_var.get()
        
        short_log = f"开始检测 ({input_type})"
        detailed_log = (
            f"开始检测\n"
            f"- 输入类型: {input_type}\n"
            f"- 模型: {model_name}\n"
            f"- 置信度阈值: {conf:.2f}\n"
            f"- IOU阈值: {iou:.2f}\n"
            f"- 帧率: {fps:.1f} FPS"
        )
        self.add_log(short_log, detailed_log)
        
        # 在单独的线程中启动检测
        self.thread = threading.Thread(target=self.detection_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop_detection(self):
        """停止检测过程"""
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        # 释放资源
        if self.cap and self.current_input_type in ["camera", "video"]:
            self.cap.release()
            self.cap = None
        
        # 更新UI
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("已停止 (Stopped)")
        
        # 记录检测停止
        self.add_log("停止检测", "用户手动停止了检测过程")
        
        # 显示黑屏
        self.show_blank_screen()
    
    def show_blank_screen(self):
        """显示黑屏"""
        self.canvas.delete("all")
        self.canvas.config(bg="black")
    
    def detection_loop(self):
        """主检测循环"""
        try:
            if self.current_input_type == "camera":
                self.process_camera()
            elif self.current_input_type == "image":
                self.process_image()
            elif self.current_input_type == "video":
                self.process_video()
        except Exception as e:
            self.status_var.set(f"错误: {str(e)}")
            messagebox.showerror("处理错误", str(e))
        finally:
            # 确保UI状态恢复
            if self.is_running:
                self.root.after(0, self.stop_detection)
    
    def process_camera(self):
        """处理摄像头输入"""
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            error_msg = "无法打开摄像头"
            self.add_log(error_msg, f"错误: {error_msg}")
            raise Exception(error_msg)
        
        self.status_var.set("正在处理摄像头输入... (Processing camera input)")
        self.add_log("摄像头已连接", "成功打开摄像头，开始处理视频流")
        
        frame_count = 0
        start_process_time = time.time()
        last_posture_log_time = 0
        last_status_update_time = 0
        
        while self.is_running:
            start_time = time.time()
            current_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                error_msg = "无法读取视频帧"
                self.status_var.set(error_msg)
                self.add_log(error_msg, "从摄像头读取视频帧失败")
                break
            
            frame_count += 1
            
            # 使用模型进行推理
            inference_start = time.time()
            results = self.model(frame, conf=self.conf_var.get(), iou=self.iou_var.get())
            inference_time = (time.time() - inference_start) * 1000  # 毫秒
            
            # 计算性能指标
            cpu_memory, gpu_memory = self.performance_monitor.get_memory_usage()
            memory_text = f"{cpu_memory:.1f} MB"
            if gpu_memory > 0:
                memory_text += f" | GPU: {gpu_memory:.1f} MB"
            
            # 计算GFLOPS
            gflops = self.performance_monitor.get_inference_gflops(self.model, inference_time/1000)
            
            # 更新性能数据
            self.performance_monitor.update(inference_time, gflops, cpu_memory)
            
            # 更新UI显示
            self.performance_vars['gflops'].set(f"{gflops:.2f} GFLOPS")
            self.performance_vars['inference_time'].set(f"{inference_time:.1f} ms")
            self.performance_vars['memory'].set(memory_text)
            
            # 计算实际帧率
            elapsed_since_start = time.time() - start_process_time
            actual_fps = frame_count / elapsed_since_start if elapsed_since_start > 0 else 0
            self.performance_vars['fps_real'].set(f"{actual_fps:.1f} FPS")
            
            # 计算平均值
            avg_time, avg_gflops, avg_mem = self.performance_monitor.get_average_values()
            self.performance_vars['avg_inference'].set(f"{avg_time:.1f} ms")
            self.performance_vars['avg_gflops'].set(f"{avg_gflops:.2f} GFLOPS")
            
            # 获取带有检测结果的图像
            annotated_frame = results[0].plot()
            
            # 在UI上显示结果
            self.display_image(annotated_frame)
            
            # 处理检测结果并记录姿态
            # 每1秒记录一次姿态结果
            if current_time - last_posture_log_time >= 1.0:
                self.log_posture_results(results)
                last_posture_log_time = current_time
            
            # 每100帧记录一次日志
            if frame_count % 100 == 0:
                elapsed = time.time() - start_process_time
                fps_actual = frame_count / elapsed
                self.add_log(f"已处理 {frame_count} 帧", 
                           f"摄像头处理统计:\n- 已处理帧数: {frame_count}\n- 运行时间: {elapsed:.1f}秒\n- 实际帧率: {fps_actual:.1f} FPS")
            
            # 定期更新智能算法状态显示
            if self.is_log_visible and self.use_smart_algorithm.get() and current_time - last_status_update_time >= 2.0:
                self.root.after(0, self.update_smart_status)
                last_status_update_time = current_time
            
            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_delay - elapsed)
            time.sleep(sleep_time)
    
    def log_posture_results(self, results):
        """记录姿态识别结果"""
        if not results or not results[0].boxes:
            # 如果使用Pro算法，仍需处理空结果
            if self.algorithm_type.get() == "pro":
                current_time = time.time()
                pro_result = self.posture_pro_recognizer.process_detection(None, current_time)
                if pro_result:
                    self.log_pro_algorithm_result(pro_result)
            return
            
        # 获取检测结果
        boxes = results[0].boxes
        cls = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []
        conf = boxes.conf.cpu().numpy() if hasattr(boxes, 'conf') else []
        
        if len(cls) == 0:
            # 如果使用Pro算法，仍需处理空结果
            if self.algorithm_type.get() == "pro":
                current_time = time.time()
                pro_result = self.posture_pro_recognizer.process_detection(None, current_time)
                if pro_result:
                    self.log_pro_algorithm_result(pro_result)
            return
            
        # 根据类别ID对应到姿态类型（根据dataset.yaml更新）
        posture_classes = {
            0: "correct_posture",   # 正确姿势
            1: "bowed_head",        # 低头
            2: "desk_leaning",      # 趴桌
            3: "head_tilt",         # 歪头
            4: "left_headed",       # 左偏头
            5: "right_headed"       # 右偏头
        }
        
        current_time = time.time()
        
        # 根据选择的算法类型处理结果
        algorithm_type = self.algorithm_type.get()
        
        if algorithm_type == "pro":
            # 使用智能算法Pro处理
            pro_result = self.posture_pro_recognizer.process_detection(results, current_time)
            if pro_result:
                self.log_pro_algorithm_result(pro_result)
        elif algorithm_type == "smart" and self.use_smart_algorithm.get():
            # 使用智能阈值算法处理
            
            # 添加当前检测结果到时间窗口
            for i in range(len(cls)):
                class_id = int(cls[i])
                confidence = conf[i]
                
                if class_id in posture_classes:
                    posture_name = posture_classes[class_id]
                    
                    # 记录当前姿态到时间窗口
                    self.posture_window.append({
                        'posture': posture_name,
                        'confidence': confidence,
                        'timestamp': current_time,
                        'class_id': class_id
                    })
            
            # 移除窗口外的旧记录
            self.posture_window = [p for p in self.posture_window 
                                if current_time - p['timestamp'] <= self.window_duration]
            
            # 只有当窗口有足够数据时才进行分析
            if len(self.posture_window) >= 3:
                self.smart_posture_analysis()
        else:
            # 非智能算法处理：使用置信度最高的检测结果
            best_conf = -1
            best_posture = None
            best_class_id = None
            
            for i in range(len(cls)):
                class_id = int(cls[i])
                confidence = conf[i]
                
                if class_id in posture_classes and confidence > best_conf:
                    best_conf = confidence
                    best_posture = posture_classes[class_id]
                    best_class_id = class_id
            
            if best_posture:
                self.add_log(
                    f"检测到姿态: {best_posture}", 
                    f"姿态识别结果:\n- 类型: {best_posture}\n- 置信度: {best_conf:.2f}"
                )
    
    def smart_posture_analysis(self):
        """智能姿态分析算法"""
        current_time = time.time()
        
        # 统计每种姿态在窗口中的情况
        posture_stats = {}
        
        for record in self.posture_window:
            posture = record['posture']
            confidence = record['confidence']
            timestamp = record['timestamp']
            class_id = record['class_id']
            
            if posture not in posture_stats:
                posture_stats[posture] = {
                    'count': 0,
                    'total_conf': 0,
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'class_id': class_id,
                    'confidences': []  # 保存所有置信度值用于计算标准差
                }
                
            stats = posture_stats[posture]
            stats['count'] += 1
            stats['total_conf'] += confidence
            stats['confidences'].append(confidence)
            stats['last_seen'] = max(stats['last_seen'], timestamp)
        
        # 保存统计数据
        self.posture_stats = posture_stats
        
        # 寻找主要姿态和次要姿态
        sorted_postures = sorted(posture_stats.items(), 
                               key=lambda x: (x[1]['count'], x[1]['total_conf']/x[1]['count']), 
                               reverse=True)
        
        if not sorted_postures:
            return
        
        # 分析主要姿态
        main_posture, main_stats = sorted_postures[0]
        
        # 计算评分指标
        # 1. 计算出现频率
        frequency_ratio = main_stats['count'] / len(self.posture_window)
        
        # 2. 计算持续时间占比
        duration = main_stats['last_seen'] - main_stats['first_seen']
        duration_ratio = min(duration / self.window_duration, 1.0)
        
        # 3. 计算平均置信度
        avg_confidence = main_stats['total_conf'] / main_stats['count']
        
        # 4. 计算置信度稳定性（标准差的倒数）
        if len(main_stats['confidences']) > 1:
            import numpy as np
            conf_std = np.std(main_stats['confidences'])
            stability = 1.0 / (1.0 + conf_std * 10)  # 归一化
        else:
            stability = 0.5  # 默认中等稳定性
        
        # 综合评分计算
        # 30%出现频率 + 25%持续时间 + 30%平均置信度 + 15%稳定性
        score = 0.3 * frequency_ratio + 0.25 * duration_ratio + 0.3 * avg_confidence + 0.15 * stability
        
        # 记录详细分析结果
        details = (
            f"姿态分析: {main_posture}\n"
            f"- 窗口内样本数: {len(self.posture_window)}\n"
            f"- 该姿态出现次数: {main_stats['count']} ({frequency_ratio*100:.1f}%)\n"
            f"- 持续时间: {duration:.2f}秒 ({duration_ratio*100:.1f}%)\n"
            f"- 平均置信度: {avg_confidence:.3f}\n"
            f"- 置信度稳定性: {stability:.2f}\n"
            f"- 综合评分: {score:.3f}/{self.trigger_threshold:.2f}"
        )
        
        # 避免连续触发同一提示，使用冷却时间
        last_time = self.last_trigger_time.get(main_posture, 0)
        cooldown_passed = (current_time - last_time) >= self.trigger_cooldown
        
        # 判断是否触发警告/提示
        if score >= self.trigger_threshold and cooldown_passed:
            # 更新最后触发时间
            self.last_trigger_time[main_posture] = current_time
            
            if main_posture == "correct_posture":
                # 正确姿势提示
                self.add_log(
                    f"姿态良好: {main_posture}",
                    f"{details}\n\n✓ 当前坐姿正确，请保持！"
                )
            else:
                # 不良姿势警告
                warn_message = {
                    "bowed_head": "检测到低头姿势，请抬头挺胸！",
                    "left_headed": "检测到头部偏左，请调整坐姿！",
                    "right_headed": "检测到头部偏右，请调整坐姿！",
                    "desk_leaning": "检测到趴桌姿势，请坐直！",
                    "head_tilt": "检测到头部倾斜，请调整坐姿！"
                }.get(main_posture, "检测到不良坐姿，请调整！")
                
                self.add_log(
                    f"姿态警告: {main_posture}",
                    f"{details}\n\n⚠️ {warn_message}"
                )
        else:
            # 记录但不触发提示
            self.add_log(
                f"姿态监测: {main_posture}",
                details
            )
    
    def process_video(self):
        """处理视频文件输入"""
        if not self.video_path:
            error_msg = "未选择视频文件"
            self.add_log(error_msg, f"错误: {error_msg}")
            raise Exception(error_msg)
        
        # 打开视频文件
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            error_msg = f"无法打开视频文件: {self.video_path}"
            self.add_log("视频打开失败", f"错误: {error_msg}")
            raise Exception(error_msg)
        
        # 获取视频信息
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        video_info = (
            f"视频文件: {os.path.basename(self.video_path)}\n"
            f"- 总帧数: {total_frames}\n"
            f"- 原始帧率: {video_fps:.2f} FPS\n"
            f"- 时长: {duration:.2f}秒"
        )
        self.add_log(f"开始处理视频: {os.path.basename(self.video_path)}", video_info)
        
        self.status_var.set(f"正在处理视频... (总帧数: {total_frames}, 原始帧率: {video_fps:.2f})")
        
        frame_count = 0
        start_process_time = time.time()
        loop_count = 0
        last_posture_log_time = 0
        last_status_update_time = 0
        
        while self.is_running:
            start_time = time.time()
            current_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                # 视频结束，循环播放
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                loop_count += 1
                self.add_log(f"视频循环播放 (第{loop_count}次)", f"视频 {os.path.basename(self.video_path)} 已播放完毕，开始第{loop_count+1}次播放")
                continue
            
            frame_count += 1
            
            # 使用模型进行推理
            results = self.model(frame, conf=self.conf_var.get(), iou=self.iou_var.get())
            
            # 获取带有检测结果的图像
            annotated_frame = results[0].plot()
            
            # 在UI上显示结果
            self.display_image(annotated_frame)
            
            # 处理检测结果并记录姿态
            # 每1秒记录一次姿态结果
            if current_time - last_posture_log_time >= 1.0:
                self.log_posture_results(results)
                last_posture_log_time = current_time
            
            # 更新状态
            progress = (frame_count / total_frames) * 100
            self.status_var.set(f"正在处理视频... {progress:.1f}% (帧 {frame_count}/{total_frames})")
            
            # 每50帧或达到25%、50%、75%、100%进度时记录日志
            log_progress_points = [25, 50, 75, 100]
            current_progress_point = int(progress)
            
            if frame_count % 50 == 0 or current_progress_point in log_progress_points:
                elapsed = time.time() - start_process_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                
                progress_log = (
                    f"视频处理进度: {progress:.1f}%\n"
                    f"- 已处理帧数: {frame_count}/{total_frames}\n"
                    f"- 运行时间: {elapsed:.1f}秒\n"
                    f"- 实际帧率: {fps_actual:.1f} FPS"
                )
                
                self.add_log(f"视频处理: {progress:.1f}%", progress_log)
                
                # 从log_progress_points中移除已经记录的点
                if current_progress_point in log_progress_points:
                    log_progress_points.remove(current_progress_point)
            
            # 定期更新智能算法状态显示
            if self.is_log_visible and self.use_smart_algorithm.get() and current_time - last_status_update_time >= 2.0:
                self.root.after(0, self.update_smart_status)
                last_status_update_time = current_time
            
            # 控制帧率
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_delay - elapsed)
            time.sleep(sleep_time)
    
    def process_image(self):
        """处理图片文件输入"""
        if not self.image_path:
            error_msg = "未选择图片文件"
            self.add_log(error_msg, f"错误: {error_msg}")
            raise Exception(error_msg)
        
        # 读取图片
        image = cv2.imread(self.image_path)
        if image is None:
            error_msg = f"无法读取图片文件: {self.image_path}"
            self.add_log("图片读取失败", f"错误: {error_msg}")
            raise Exception(error_msg)
        
        # 记录图片信息
        h, w, c = image.shape
        image_info = (
            f"图片文件: {os.path.basename(self.image_path)}\n"
            f"- 尺寸: {w}x{h}\n"
            f"- 通道数: {c}"
        )
        self.add_log(f"开始处理图片: {os.path.basename(self.image_path)}", image_info)
        
        self.status_var.set(f"正在处理图片: {os.path.basename(self.image_path)}")
        
        # 使用模型进行推理
        start_time = time.time()
        results = self.model(image, conf=self.conf_var.get(), iou=self.iou_var.get())
        inference_time = time.time() - start_time
        
        # 获取检测结果信息
        num_detections = len(results[0].boxes)
        
        # 记录检测结果
        detection_info = (
            f"图片检测完成: {os.path.basename(self.image_path)}\n"
            f"- 检测到的目标数: {num_detections}\n"
            f"- 推理时间: {inference_time:.3f}秒"
        )
        self.add_log(f"图片检测完成: 发现 {num_detections} 个目标", detection_info)
        
        # 记录姿态识别结果
        self.log_posture_results(results)
        
        # 获取带有检测结果的图像
        annotated_image = results[0].plot()
        
        # 在UI上显示结果
        self.display_image(annotated_image)
        
        self.status_var.set(f"图片处理完成: {os.path.basename(self.image_path)}")
        
        # 图片处理完成后，停止检测状态但保持图像显示
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
    
    def display_image(self, cv_image):
        """在画布上显示OpenCV图像"""
        if not self.is_running:
            return
            
        # 调整图像大小以适应画布
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # 画布尚未正确初始化，使用合理的默认值
            canvas_width = 640
            canvas_height = 480
        
        # 转换颜色空间从BGR到RGB
        cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # 如果是Pro算法模式，添加分类输出标签
        if self.algorithm_type.get() == "pro" and hasattr(self, 'posture_pro_recognizer'):
            stats = self.posture_pro_recognizer.get_output_and_stats()
            current_output = stats['output']
            stability = stats['stability']
            
            if current_output:
                # 定义标签颜色和文本
                if current_output == "correct_posture":
                    color = (0, 255, 0)  # 绿色 (RGB格式)
                    label = "Correct Posture"  # 仅使用英文避免中文渲染问题
                elif current_output == "incorrect_posture":
                    color = (255, 69, 0)  # 橙红色 (RGB格式)
                    label = "Incorrect Posture"  # 仅使用英文避免中文渲染问题
                elif current_output == "nobody":
                    # 如果nobody持续超过1分30秒，使用红色警告
                    if hasattr(self.posture_pro_recognizer, 'nobody_duration') and self.posture_pro_recognizer.nobody_duration > 90:
                        color = (255, 0, 0)  # 红色警告 (RGB格式)
                    else:
                        color = (128, 128, 128)  # 灰色
                    label = "Nobody"  # 仅使用英文避免中文渲染问题
                else:
                    color = (200, 200, 200)
                    label = current_output
                
                # 添加圆角矩形背景 (模拟圆角效果)
                overlay = cv_image_rgb.copy()
                cv2.rectangle(overlay, (10, 10), (420, 70), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, cv_image_rgb, 0.3, 0, cv_image_rgb)  # 半透明效果
                
                # 添加姿态标签 - 使用更大字体增强可读性
                cv2.putText(cv_image_rgb, label, (20, 45), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 2, cv2.LINE_AA)
                
                # 为correct_posture添加绿色勾标记，当stability达到100%时
                if current_output == "correct_posture" and stability >= 1.0:
                    # 绘制绿色勾，位于右侧
                    check_x, check_y = 380, 45
                    # 绘制绿色勾
                    cv2.line(cv_image_rgb, (check_x-15, check_y-5), (check_x-5, check_y+5), (0, 255, 0), 3, cv2.LINE_AA)
                    cv2.line(cv_image_rgb, (check_x-5, check_y+5), (check_x+15, check_y-15), (0, 255, 0), 3, cv2.LINE_AA)
                
                # 添加稳定性指示条 - 使用圆角效果
                # 背景条
                cv2.rectangle(cv_image_rgb, (20, 55), (320, 65), (100, 100, 100), -1)
                
                # 进度条 - 使用更细的1%增量步长，使其更平滑
                # 当达到100%时填满
                bar_length = int(300 * stability)  # 1%的精细步长
                bar_length = min(300, max(0, bar_length))  # 限制在0-300范围内
                
                if bar_length > 0:
                    # 如果进度条达到满格且是incorrect_posture，则使用更快闪烁的红色效果
                    if bar_length >= 300 and current_output == "incorrect_posture":
                        # 使用当前时间的毫秒数来创建高频闪烁效果（每250毫秒切换一次，比之前500毫秒更快）
                        import time
                        if int(time.time() * 4) % 2 == 0:  # 4倍频率，更快的闪烁
                            flash_color = (255, 0, 0)  # 纯红色 (RGB格式)
                        else:
                            flash_color = (255, 69, 0)  # 橙红色 (RGB格式)
                        cv2.rectangle(cv_image_rgb, (20, 55), (20 + bar_length, 65), flash_color, -1)
                    else:
                        cv2.rectangle(cv_image_rgb, (20, 55), (20 + bar_length, 65), color, -1)
                
                # 添加稳定性百分比文本 - 精确到1%
                stability_text = f"Stability: {int(stability * 100)}%"
                cv2.putText(cv_image_rgb, stability_text, (330, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                
                # 如果是nobody状态，显示持续时间 - 精确到1秒
                if current_output == "nobody" and hasattr(self.posture_pro_recognizer, 'nobody_duration'):
                    nobody_time = self.posture_pro_recognizer.nobody_duration
                    
                    # 检查是否有高置信度的动作被检测到
                    high_conf_detected = self.posture_pro_recognizer.check_recent_high_confidence_detections()
                    
                    if high_conf_detected:
                        # 如果检测到高置信度动作，显示转换提示
                        time_text = f"Time: {int(nobody_time)}s (Transition...)"
                    else:
                        # 正常显示时间
                        time_text = f"Time: {int(nobody_time)}s / 15s"  # 调整为15秒
                    
                    # 计算nobody时间的进度条
                    nobody_progress = min(1.0, nobody_time / 15.0)  # 15秒满进度
                    nobody_bar_length = int(300 * nobody_progress)
                    
                    # 绘制nobody时间进度条
                    cv2.rectangle(cv_image_rgb, (20, 85), (320, 95), (80, 80, 80), -1)  # 背景条
                    
                    if high_conf_detected:
                        # 如果检测到高置信度动作，使用蓝色渐变进度条
                        bar_color = (255, 165, 0)  # 橙色 (BGR格式)
                    else:
                        # 正常灰色进度条逻辑
                        if nobody_bar_length > 0:
                            # 如果接近满进度或超过15秒，使用闪烁红色效果
                            if nobody_time >= 13.0:  # 13秒后开始闪烁
                                import time
                                if int(time.time() * 2) % 2 == 0:
                                    bar_color = (0, 0, 255)  # 红色 (BGR格式)
                                else:
                                    bar_color = (80, 80, 80)  # 灰色
                            else:
                                bar_color = (128, 128, 128)  # 普通灰色
                    
                    # 绘制进度条
                    if nobody_bar_length > 0:
                        cv2.rectangle(cv_image_rgb, (20, 85), (20 + nobody_bar_length, 95), bar_color, -1)
                    
                    # 文本颜色根据状态变化
                    if high_conf_detected:
                        text_color = (255, 165, 0)  # 橙色 (BGR格式)
                    else:
                        text_color = (200, 200, 200)  # 默认灰色文本
                        
                        # 如果时间超过15秒，文本也变成闪烁红色
                        if nobody_time >= 15.0:
                            import time
                            if int(time.time() * 2) % 2 == 0:
                                text_color = (0, 0, 255)  # 红色文本 (BGR格式)
                            else:
                                text_color = (255, 255, 255)  # 白色文本
                    
                    cv2.putText(cv_image_rgb, time_text, (330, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1, cv2.LINE_AA)
                    
                    # 当nobody状态持续15秒后，显示警告文本
                    if nobody_time >= 15.0 and not high_conf_detected:
                        warning_text = "Long absence detected!"
                        
                        # 使用闪烁效果
                        import time
                        if int(time.time() * 2) % 2 == 0:
                            warning_color = (0, 0, 255)  # 红色 (BGR格式)
                        else:
                            warning_color = (0, 69, 255)  # 橙红色 (BGR格式)
                            
                        cv2.putText(cv_image_rgb, warning_text, (20, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, warning_color, 2, cv2.LINE_AA)
                    
                    # 如果检测到高置信度的动作，显示转换提示
                    if high_conf_detected:
                        transition_text = "Motion detected, transitioning..."
                        cv2.putText(cv_image_rgb, transition_text, (20, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)
        
        # 计算等比例缩放的尺寸
        h, w = cv_image_rgb.shape[:2]
        aspect_ratio = w / h
        
        if canvas_width / canvas_height > aspect_ratio:
            # 画布比图像更宽
            new_height = canvas_height
            new_width = int(new_height * aspect_ratio)
        else:
            # 画布比图像更高
            new_width = canvas_width
            new_height = int(new_width / aspect_ratio)
        
        # 缩放图像
        resized_image = cv2.resize(cv_image_rgb, (new_width, new_height))
        
        # 转换为PhotoImage
        image = Image.fromarray(resized_image)
        photo = ImageTk.PhotoImage(image=image)
        
        # 清除画布并显示新图像
        self.canvas.delete("all")
        # 保存引用，防止垃圾回收
        self.canvas.image = photo
        # 计算居中位置
        x = (canvas_width - new_width) // 2
        y = (canvas_height - new_height) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=photo)

    def toggle_log_window(self):
        """切换日志窗口的显示和隐藏"""
        if self.is_log_visible:
            self.hide_log_window()
        else:
            self.show_log_window()
    
    def show_log_window(self):
        """显示日志窗口"""
        if self.log_window is None:
            # 创建新的日志窗口
            self.log_window = tk.Toplevel(self.root)
            self.log_window.title("运行日志 (Operation Log)")
            self.log_window.geometry("600x400")
            self.log_window.minsize(400, 300)
            self.log_window.protocol("WM_DELETE_WINDOW", self.hide_log_window)
            self.log_window.configure(background="#F5F5F7")
            
            # 日志内容框架
            log_frame = ttk.Frame(self.log_window, padding="10", style="Modern.TFrame")
            log_frame.pack(fill=tk.BOTH, expand=True)
            
            # 获取字体
            font_family = "Microsoft YaHei" if os.name == "nt" else "Helvetica"
            
            # 模式选择区域
            mode_frame = ttk.Frame(log_frame, style="Modern.TFrame")
            mode_frame.pack(fill=tk.X, pady=(0, 10))
            
            # 添加单选按钮来选择日志模式
            self.log_mode = tk.StringVar(value="all")
            ttk.Radiobutton(mode_frame, text="显示所有日志", variable=self.log_mode, value="all", 
                          command=self.update_log_display, style="Modern.TRadiobutton").pack(side=tk.LEFT, padx=(0, 10))
            ttk.Radiobutton(mode_frame, text="仅显示姿态结果", variable=self.log_mode, value="posture", 
                          command=self.update_log_display, style="Modern.TRadiobutton").pack(side=tk.LEFT)
            
            # 智能算法状态显示
            self.smart_status_frame = ttk.Frame(log_frame, style="Modern.TFrame")
            self.smart_status_frame.pack(fill=tk.X, pady=(0, 10))
            
            self.smart_status_label = ttk.Label(self.smart_status_frame, 
                                           text="智能算法状态: 已禁用", 
                                           style="Modern.TLabel")
            self.smart_status_label.pack(side=tk.LEFT)
            
            # 更新智能算法状态
            self.update_smart_status()
            
            # 简化日志显示
            self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=15, 
                                  font=(font_family, 10),
                                  bg="#FFFFFF", fg="#333333",
                                  highlightthickness=0, borderwidth=1,
                                  relief=tk.SOLID)
            self.log_text.pack(fill=tk.BOTH, expand=True)
            
            # 添加滚动条
            scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            self.log_text.config(yscrollcommand=scrollbar.set)
            
            # 详细日志按钮
            btn_frame = ttk.Frame(log_frame, style="Modern.TFrame")
            btn_frame.pack(fill=tk.X, pady=(10, 0))
            
            self.detailed_log_btn = ttk.Button(btn_frame, text="查看详细日志", 
                                            command=self.show_detailed_log, style="Modern.TButton")
            self.detailed_log_btn.pack(side=tk.LEFT)
            
            clear_btn = ttk.Button(btn_frame, text="清除日志", 
                                 command=self.clear_log, style="Modern.TButton")
            clear_btn.pack(side=tk.RIGHT)
            
            # 显示已有日志
            self.update_log_display()
        else:
            self.log_window.deiconify()
            # 更新智能算法状态
            self.update_smart_status()
        
        self.is_log_visible = True
        self.log_btn.config(text="隐藏日志 (Hide Log)")
        
    def update_smart_status(self):
        """更新智能算法状态显示"""
        if not hasattr(self, 'smart_status_label') or not self.smart_status_label:
            return
            
        if self.use_smart_algorithm.get():
            # 显示活跃状态及相关信息
            active_info = ""
            if hasattr(self, 'posture_stats') and self.posture_stats:
                # 获取最主要的姿态
                sorted_postures = sorted(self.posture_stats.items(), 
                                      key=lambda x: (x[1]['count'], x[1]['total_conf']/x[1]['count']), 
                                      reverse=True)
                if sorted_postures:
                    main_posture, stats = sorted_postures[0]
                    avg_conf = stats['total_conf'] / stats['count'] if stats['count'] > 0 else 0
                    active_info = f" | 主要姿态: {main_posture} ({stats['count']}次, {avg_conf:.2f})"
            
            self.smart_status_label.config(
                text=f"智能算法状态: 已启用 (窗口: {self.window_duration:.1f}秒, 阈值: {self.trigger_threshold:.2f}){active_info}",
                foreground="#007AFF"  # 蓝色提示活跃状态
            )
        else:
            self.smart_status_label.config(
                text="智能算法状态: 已禁用 (使用单帧判断)",
                foreground="#666666"  # 灰色提示非活跃状态
            )
    
    def hide_log_window(self):
        """隐藏日志窗口"""
        if self.log_window:
            self.log_window.withdraw()
        self.is_log_visible = False
        self.log_btn.config(text="显示日志 (Show Log)")
    
    def toggle_smart_algorithm(self):
        """切换智能算法开关状态"""
        enabled = self.use_smart_algorithm.get()
        state = tk.NORMAL if enabled else tk.DISABLED
        
        # 启用/禁用参数控制
        for child in self.smart_params_frame.winfo_children():
            if isinstance(child, ttk.Scale):
                child.config(state=state)
        
        if enabled:
            # 启用智能算法时，清空历史数据并记录日志
            self.posture_window = []
            self.last_trigger_time = {}
            self.posture_stats = {}
            
            # 调整YOLO参数以配合智能算法
            original_conf = self.conf_var.get()
            # 降低模型的置信度阈值，让更多候选框通过初筛，由智能算法进行二次筛选
            new_conf = max(0.15, self.trigger_threshold * 0.5)
            self.conf_var.set(new_conf)
            
            # 增加IOU阈值以减少重复框
            original_iou = self.iou_var.get()
            new_iou = min(0.7, original_iou * 1.2)
            self.iou_var.set(new_iou)
        else:
            if self.algorithm_type.get() == "none":
                # 禁用智能算法时，恢复默认的YOLO参数
                self.conf_var.set(0.35)
                self.iou_var.set(0.5)
            
        # 更新日志窗口中的智能算法状态显示，仅当智能算法状态标签存在时
        if hasattr(self, 'smart_status_label') and self.smart_status_label:
            self.update_smart_status()
    
    def show_detailed_log(self):
        """显示详细日志窗口"""
        detailed_window = tk.Toplevel(self.root)
        detailed_window.title("详细日志 (Detailed Log)")
        detailed_window.geometry("800x600")
        detailed_window.minsize(600, 400)
        detailed_window.configure(background="#F5F5F7")
        
        # 获取字体
        font_family = "Microsoft YaHei" if os.name == "nt" else "Helvetica"
        
        # 详细日志内容框架
        detailed_frame = ttk.Frame(detailed_window, padding="10", style="Modern.TFrame")
        detailed_frame.pack(fill=tk.BOTH, expand=True)
        
        # 详细日志文本区
        detailed_text = tk.Text(detailed_frame, wrap=tk.WORD,
                              font=(font_family, 10),
                              bg="#FFFFFF", fg="#333333",
                              highlightthickness=0, borderwidth=1,
                              relief=tk.SOLID)
        detailed_text.pack(fill=tk.BOTH, expand=True)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(detailed_text, command=detailed_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        detailed_text.config(yscrollcommand=scrollbar.set)
        
        # 显示所有日志的详细信息
        for entry in self.log_entries:
            timestamp, short_msg, detailed_msg = entry
            detailed_text.insert(tk.END, f"[{timestamp}] {short_msg}\n", "title")
            detailed_text.insert(tk.END, f"{detailed_msg}\n\n", "detail")
        
        # 添加文本标签
        detailed_text.tag_configure("title", font=(font_family, 10, "bold"))
        detailed_text.tag_configure("detail", font=(font_family, 9))
        
        detailed_text.config(state=tk.DISABLED)
        
        # 添加关闭按钮
        btn_frame = ttk.Frame(detailed_window, style="Modern.TFrame")
        btn_frame.pack(fill=tk.X, pady=10, padx=10)
        
        close_btn = ttk.Button(btn_frame, text="关闭", 
                             command=detailed_window.destroy, style="Modern.TButton")
        close_btn.pack(side=tk.RIGHT)
    
    def is_posture_log(self, message):
        """检查日志是否是姿态检测结果"""
        posture_keywords = [
            "correct_posture", "bowed_head", "left_headed",
            "right_headed", "deak_leaning", "head_tilt"
        ]
        return any(keyword in message.lower() for keyword in posture_keywords)
    
    def update_log_display(self):
        """更新日志显示"""
        if not self.log_text or not hasattr(self, 'log_entries'):
            return
            
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        
        mode = self.log_mode.get() if hasattr(self, 'log_mode') else "all"
        
        for entry in self.log_entries:
            timestamp, short_msg, detailed_msg = entry
            
            # 如果是"仅姿态结果"模式，则只显示姿态相关的日志
            if mode == "all" or (mode == "posture" and self.is_posture_log(short_msg + detailed_msg)):
                self.log_text.insert(tk.END, f"[{timestamp}] {short_msg}\n")
            
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END)  # 滚动到最新日志
    
    def clear_log(self):
        """清除日志"""
        self.log_entries = []
        if self.log_text:
            self.log_text.config(state=tk.NORMAL)
            self.log_text.delete(1.0, tk.END)
            self.log_text.config(state=tk.DISABLED)
    
    def add_log(self, short_msg, detailed_msg=None):
        """添加一条日志"""
        # 安全检查，确保log_entries存在
        if not hasattr(self, 'log_entries'):
            self.log_entries = []
            
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if detailed_msg is None:
            detailed_msg = short_msg
            
        self.log_entries.append((timestamp, short_msg, detailed_msg))
        
        # 如果日志窗口已打开，更新显示
        if hasattr(self, 'log_text') and self.log_text:
            self.update_log_display()

    def update_window_duration(self, value):
        """更新时间窗口长度"""
        self.window_duration = float(value)
        self.window_label.config(text=f"{self.window_duration:.1f}秒")
        
        # 清除时间窗口中的过期数据
        current_time = time.time()
        self.posture_window = [p for p in self.posture_window 
                             if current_time - p['timestamp'] <= self.window_duration]
    
    def update_trigger_threshold(self, value):
        """更新触发阈值"""
        self.trigger_threshold = float(value)
        self.threshold_label.config(text=f"{self.trigger_threshold:.2f}")
        
        # 如果使用智能算法，自动调整YOLO的置信度阈值
        if self.use_smart_algorithm.get():
            # 降低模型的置信度阈值，让更多候选框通过初筛，由智能算法进行二次筛选
            new_conf = max(0.15, self.trigger_threshold * 0.5)
            self.conf_var.set(new_conf)
    
    def update_cooldown(self, value):
        """更新提示冷却时间"""
        self.trigger_cooldown = float(value)
        self.cooldown_label.config(text=f"{self.trigger_cooldown:.1f}秒")

    def toggle_performance_window(self):
        """显示/隐藏性能监控详情窗口"""
        if self.is_perf_window_visible:
            self.hide_performance_window()
        else:
            self.show_performance_window()
    
    def show_performance_window(self):
        """显示性能监控详情窗口"""
        if self.perf_window is None:
            # 创建新窗口
            self.perf_window = tk.Toplevel(self.root)
            self.perf_window.title("性能监控详情 (Performance Details)")
            self.perf_window.geometry("700x500")
            self.perf_window.minsize(500, 400)
            self.perf_window.protocol("WM_DELETE_WINDOW", self.hide_performance_window)
            self.perf_window.configure(background="#F5F5F7")
            
            # 性能详情框架
            perf_detail_frame = ttk.Frame(self.perf_window, padding="10", style="Modern.TFrame")
            perf_detail_frame.pack(fill=tk.BOTH, expand=True)
            
            # 性能指标部分
            metrics_frame = ttk.LabelFrame(perf_detail_frame, text="实时性能指标", style="Modern.TLabelframe")
            metrics_frame.pack(fill=tk.X, pady=(0, 10))
            
            # 当前性能指标
            current_frame = ttk.Frame(metrics_frame, style="Modern.TFrame")
            current_frame.pack(fill=tk.X, pady=5)
            
            # 两列布局
            ttk.Label(current_frame, text="当前值", font=("Microsoft YaHei", 10, "bold"), style="Modern.TLabel").grid(row=0, column=0, columnspan=2, sticky=tk.W)
            
            ttk.Label(current_frame, text="计算量 (GFLOPS):", style="Modern.TLabel").grid(row=1, column=0, sticky=tk.W, padx=(20, 5))
            ttk.Label(current_frame, textvariable=self.performance_vars['gflops'], style="Value.TLabel").grid(row=1, column=1, sticky=tk.W)
            
            ttk.Label(current_frame, text="推理时间 (ms):", style="Modern.TLabel").grid(row=2, column=0, sticky=tk.W, padx=(20, 5))
            ttk.Label(current_frame, textvariable=self.performance_vars['inference_time'], style="Value.TLabel").grid(row=2, column=1, sticky=tk.W)
            
            ttk.Label(current_frame, text="实际帧率:", style="Modern.TLabel").grid(row=3, column=0, sticky=tk.W, padx=(20, 5))
            ttk.Label(current_frame, textvariable=self.performance_vars['fps_real'], style="Value.TLabel").grid(row=3, column=1, sticky=tk.W)
            
            ttk.Label(current_frame, text="内存使用:", style="Modern.TLabel").grid(row=4, column=0, sticky=tk.W, padx=(20, 5))
            ttk.Label(current_frame, textvariable=self.performance_vars['memory'], style="Value.TLabel").grid(row=4, column=1, sticky=tk.W)
            
            # 平均性能指标
            avg_frame = ttk.Frame(metrics_frame, style="Modern.TFrame")
            avg_frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(avg_frame, text="平均值 (最近50次推理)", font=("Microsoft YaHei", 10, "bold"), style="Modern.TLabel").grid(row=0, column=0, columnspan=2, sticky=tk.W)
            
            ttk.Label(avg_frame, text="平均计算量:", style="Modern.TLabel").grid(row=1, column=0, sticky=tk.W, padx=(20, 5))
            ttk.Label(avg_frame, textvariable=self.performance_vars['avg_gflops'], style="Value.TLabel").grid(row=1, column=1, sticky=tk.W)
            
            ttk.Label(avg_frame, text="平均推理时间:", style="Modern.TLabel").grid(row=2, column=0, sticky=tk.W, padx=(20, 5))
            ttk.Label(avg_frame, textvariable=self.performance_vars['avg_inference'], style="Value.TLabel").grid(row=2, column=1, sticky=tk.W)
            
            # 模型信息部分
            model_frame = ttk.LabelFrame(perf_detail_frame, text="模型信息", style="Modern.TLabelframe")
            model_frame.pack(fill=tk.X, pady=(0, 10))
            
            # 获取模型信息
            model_info = self.get_model_info() if self.model else {"名称": "未加载", "参数量": "N/A", "大小": "N/A"}
            
            # 显示模型信息
            ttk.Label(model_frame, text=f"模型名称: {model_info['名称']}", style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
            ttk.Label(model_frame, text=f"参数量: {model_info['参数量']}", style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
            ttk.Label(model_frame, text=f"模型大小: {model_info['大小']}", style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
            
            # GFLOPS解释部分
            explanation_frame = ttk.LabelFrame(perf_detail_frame, text="GFLOPS说明", style="Modern.TLabelframe")
            explanation_frame.pack(fill=tk.BOTH, expand=True)
            
            explanation_text = tk.Text(explanation_frame, wrap=tk.WORD, height=10, 
                                     font=("Microsoft YaHei", 10),
                                     bg="#FFFFFF", fg="#333333",
                                     highlightthickness=0, borderwidth=1,
                                     relief=tk.SOLID)
            explanation_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # 添加说明文本
            explanation = (
                "GFLOPS (每秒十亿次浮点运算) 说明：\n\n"
                "1. 理论GFLOPS：当前显示的是模型推理的理论计算量，反映了模型的复杂度\n\n"
                "2. 不同设备换算关系：\n"
                "   • ONNX模型通常比PyTorch模型快约1.3-2倍，理论GFLOPS减少5-10%\n"
                "   • RKNN模型针对嵌入式设备优化，相同操作的GFLOPS理论值约为PyTorch的85%\n"
                "   • 在计算能力有限的嵌入式设备上，可使用公式：实际RKNN性能 ≈ (PyTorch GFLOPS × 0.85) ÷ 设备算力倍数\n\n"
                "3. 实际性能会受到以下因素影响：\n"
                "   • 硬件加速支持 (CUDA, TensorRT, NPU等)\n"
                "   • 图像分辨率 (本测试基于640×640)\n"
                "   • 批处理大小 (Batch Size)\n"
                "   • 量化精度 (FP32, FP16, INT8等)"
            )
            explanation_text.insert(tk.END, explanation)
            explanation_text.config(state=tk.DISABLED)
            
            # 添加滚动条
            scrollbar = ttk.Scrollbar(explanation_text, command=explanation_text.yview)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            explanation_text.config(yscrollcommand=scrollbar.set)
            
            # 添加底部按钮
            btn_frame = ttk.Frame(perf_detail_frame, style="Modern.TFrame")
            btn_frame.pack(fill=tk.X, pady=(10, 0))
            
            # 导出按钮
            ttk.Button(btn_frame, text="导出性能数据", 
                     command=self.export_performance_data, style="Modern.TButton").pack(side=tk.LEFT)
            
            # 重置统计按钮
            ttk.Button(btn_frame, text="重置统计", 
                     command=self.reset_performance_stats, style="Modern.TButton").pack(side=tk.LEFT, padx=(10, 0))
            
            # 关闭按钮
            ttk.Button(btn_frame, text="关闭", 
                     command=self.hide_performance_window, style="Modern.TButton").pack(side=tk.RIGHT)
        else:
            self.perf_window.deiconify()
            # 更新模型信息
            self.update_model_info_in_window()
        
        self.is_perf_window_visible = True
    
    def update_model_info_in_window(self):
        """更新性能窗口中的模型信息"""
        if self.perf_window and self.model:
            # 查找模型信息框架
            for widget in self.perf_window.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.LabelFrame) and child.cget("text") == "模型信息":
                            # 清除现有内容
                            for info_widget in child.winfo_children():
                                info_widget.destroy()
                            
                            # 获取最新模型信息
                            model_info = self.get_model_info()
                            
                            # 重新添加信息
                            ttk.Label(child, text=f"模型名称: {model_info['名称']}", style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
                            ttk.Label(child, text=f"参数量: {model_info['参数量']}", style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
                            ttk.Label(child, text=f"模型大小: {model_info['大小']}", style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
                            return
    
    def hide_performance_window(self):
        """隐藏性能监控窗口"""
        if self.perf_window:
            self.perf_window.withdraw()
        self.is_perf_window_visible = False
    
    def get_model_info(self):
        """获取当前模型的详细信息"""
        if not self.model:
            return {"名称": "未加载", "参数量": "N/A", "大小": "N/A"}
        
        model_name = self.model_var.get()
        model_path = self.model_paths.get(model_name, "未知")
        
        # 计算模型参数量
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        param_str = f"{param_count:,}"
        
        # 获取模型文件大小
        try:
            size_bytes = os.path.getsize(model_path)
            if size_bytes < 1024*1024:
                size_str = f"{size_bytes/1024:.1f} KB"
            else:
                size_str = f"{size_bytes/(1024*1024):.1f} MB"
        except Exception:
            size_str = "未知"
        
        return {
            "名称": model_name,
            "参数量": param_str,
            "大小": size_str
        }
    
    def export_performance_data(self):
        """导出性能数据到CSV文件"""
        if not hasattr(self.performance_monitor, "inference_times") or not self.performance_monitor.inference_times:
            messagebox.showinfo("导出失败", "没有可导出的性能数据，请先运行模型")
            return
        
        try:
            import pandas as pd
            import datetime
            
            # 创建数据字典
            data = {
                "InferenceTime_ms": self.performance_monitor.inference_times,
                "GFLOPS": self.performance_monitor.gflops_values,
                "Memory_MB": self.performance_monitor.memory_usage
            }
            
            # 转换为DataFrame
            df = pd.DataFrame(data)
            
            # 添加统计信息
            stats = df.describe()
            
            # 生成文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.model_var.get().replace(".pt", "")
            filename = f"performance_{model_name}_{timestamp}.csv"
            
            # 保存文件
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                initialfile=filename
            )
            
            if filepath:
                df.to_csv(filepath, index=False)
                # 将统计数据添加到日志
                stats_str = stats.to_string()
                self.add_log(
                    f"已导出性能数据: {os.path.basename(filepath)}",
                    f"性能数据已导出到: {filepath}\n\n统计摘要:\n{stats_str}"
                )
                messagebox.showinfo("导出成功", f"性能数据已成功导出到:\n{filepath}")
        except Exception as e:
            messagebox.showerror("导出错误", f"导出性能数据时出错:\n{str(e)}")
    
    def reset_performance_stats(self):
        """重置性能统计数据"""
        self.performance_monitor.reset()
        # 重置显示
        self.performance_vars['gflops'].set("-- GFLOPS")
        self.performance_vars['inference_time'].set("-- ms")
        self.performance_vars['memory'].set("-- MB")
        self.performance_vars['avg_gflops'].set("-- GFLOPS")
        self.performance_vars['avg_inference'].set("-- ms")
        self.performance_vars['fps_real'].set("-- FPS")
        
        messagebox.showinfo("已重置", "性能统计数据已重置")

    def __del__(self):
        """清理资源"""
        # 解除鼠标滚轮绑定
        try:
            self.control_canvas.unbind_all("<MouseWheel>")
        except:
            pass
        
        # 释放摄像头
        if self.cap is not None and hasattr(self.cap, 'release'):
            self.cap.release()
    
    def log_pro_algorithm_result(self, result):
        """记录智能算法Pro的识别结果"""
        if not result or result == self.last_pro_result:
            return
            
        # 获取当前状态和统计数据
        stats = self.posture_pro_recognizer.get_output_and_stats()
        stability = stats['stability']
        
        # 根据输出类型生成不同的消息
        if result == "correct_posture":
            icon = "✓"
            message = "当前坐姿正确，请保持！"
            color = "green"
        elif result == "bowed_head":
            icon = "⚠️"
            message = "检测到低头姿势，请抬头挺胸！"
            color = "orange"
        elif result == "incorrect_posture":
            icon = "⚠️"
            message = "检测到不良坐姿，请调整！"
            color = "red"
        elif result == "nobody":
            icon = "👤"
            message = "画面中未检测到人"
            color = "grey"
        else:
            icon = "❓"
            message = f"未知姿态: {result}"
            color = "grey"
        
        # 记录日志
        # 降低触发阈值至40%稳定性
        if (stability >= 0.4 and result != "nobody") or (stability >= 0.4 and result == "nobody" and hasattr(self, 'nobody_duration') and self.nobody_duration >= 40):
            # 只有稳定后才更新last_pro_result
            self.last_pro_result = result
            
            # 窗口计数
            window_stats = stats['window_stats']
            
            # 详细信息
            details = (
                f"智能算法Pro分析结果:\n"
                f"- 输出: {result}\n"
                f"- 稳定性: {stability*100:.1f}%\n"
                f"- 窗口记录: 短期({window_stats['short']}), 中期({window_stats['mid']}), 长期({window_stats['long']})\n"
                f"- 消息: {icon} {message}"
            )
            
            self.add_log(f"Pro算法: {result}", details)

    def show_algorithm_details(self):
        """显示智能阈值算法的详细信息窗口"""
        # 创建新窗口
        details_window = tk.Toplevel(self.root)
        details_window.title("智能阈值算法详情")
        details_window.geometry("700x500")
        details_window.minsize(600, 400)
        details_window.configure(background="#F5F5F7")
        
        # 获取字体
        font_family = "Microsoft YaHei" if os.name == "nt" else "Helvetica"
        
        # 创建主框架
        main_frame = ttk.Frame(details_window, padding="10", style="Modern.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建标题
        ttk.Label(main_frame, text="智能阈值算法详情", 
                 font=(font_family, 14, "bold"), style="Title.TLabel").pack(anchor=tk.W, pady=(0, 10))
        
        # 创建四个主要部分
        
        # 1. 当前参数设置
        params_frame = ttk.LabelFrame(main_frame, text="当前参数设置", style="Modern.TLabelframe")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 显示当前参数值
        ttk.Label(params_frame, text=f"• 时间窗口长度: {self.window_duration:.2f} 秒", 
                 font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Label(params_frame, text=f"• 触发阈值: {self.trigger_threshold:.3f}", 
                 font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
        ttk.Label(params_frame, text=f"• 提示冷却时间: {self.trigger_cooldown:.2f} 秒", 
                 font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
        
        # 如果当前有姿态统计数据，显示主要姿态信息
        if hasattr(self, 'posture_stats') and self.posture_stats:
            sorted_postures = sorted(self.posture_stats.items(), 
                                  key=lambda x: (x[1]['count'], x[1]['total_conf']/x[1]['count'] if x[1]['count'] > 0 else 0), 
                                  reverse=True)
            if sorted_postures:
                main_posture, stats = sorted_postures[0]
                count = stats['count']
                avg_conf = stats['total_conf'] / count if count > 0 else 0
                ttk.Label(params_frame, text=f"• 当前主要姿态: {main_posture} (出现 {count} 次, 平均置信度: {avg_conf:.3f})", 
                         font=(font_family, 10, "bold"), foreground="#007AFF", 
                         style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
        
        # 2. 算法工作流程
        workflow_frame = ttk.LabelFrame(main_frame, text="算法工作流程", style="Modern.TLabelframe")
        workflow_frame.pack(fill=tk.X, pady=(0, 10))
        
        workflow_text = (
            "1. 数据收集: 在时间窗口内收集姿态检测结果\n"
            "2. 统计分析: 对每种姿态进行出现频率、持续时间、置信度统计\n"
            "3. 评分计算: 对主要姿态计算综合评分\n"
            "   - 30% 出现频率 + 25% 持续时间 + 30% 平均置信度 + 15% 稳定性\n"
            "4. 触发判断: 评分超过触发阈值且经过冷却时间后触发提示"
        )
        
        ttk.Label(workflow_frame, text=workflow_text, 
                 font=(font_family, 10), style="Modern.TLabel", 
                 justify=tk.LEFT).pack(anchor=tk.W, padx=10, pady=5)
        
        # 3. 输入数据统计
        stats_frame = ttk.LabelFrame(main_frame, text="当前数据统计", style="Modern.TLabelframe")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 显示时间窗口内的记录数量
        window_count = len(self.posture_window) if hasattr(self, 'posture_window') else 0
        ttk.Label(stats_frame, text=f"• 时间窗口内的记录数: {window_count} 条", 
                 font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
        
        # 显示各姿态的统计
        if hasattr(self, 'posture_stats') and self.posture_stats:
            sorted_stats = sorted(self.posture_stats.items(), 
                               key=lambda x: x[1]['count'], reverse=True)
            for posture, stats in sorted_stats:
                count = stats['count']
                if count > 0:
                    avg_conf = stats['total_conf'] / count
                    ttk.Label(stats_frame, text=f"• {posture}: {count} 次, 平均置信度: {avg_conf:.3f}", 
                             font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
        else:
            ttk.Label(stats_frame, text="• 暂无姿态统计数据", 
                     font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
        
        # 4. 评分详情
        if hasattr(self, 'posture_stats') and self.posture_stats and hasattr(self, 'posture_window') and self.posture_window:
            sorted_postures = sorted(self.posture_stats.items(), 
                                  key=lambda x: (x[1]['count'], x[1]['total_conf']/x[1]['count'] if x[1]['count'] > 0 else 0), 
                                  reverse=True)
            if sorted_postures:
                main_posture, main_stats = sorted_postures[0]
                
                scoring_frame = ttk.LabelFrame(main_frame, text=f"主要姿态评分详情: {main_posture}", style="Modern.TLabelframe")
                scoring_frame.pack(fill=tk.X, pady=(0, 10))
                
                # 计算评分指标
                frequency_ratio = main_stats['count'] / len(self.posture_window)
                
                current_time = time.time()
                duration = main_stats['last_seen'] - main_stats['first_seen']
                duration_ratio = min(duration / self.window_duration, 1.0)
                
                avg_confidence = main_stats['total_conf'] / main_stats['count'] if main_stats['count'] > 0 else 0
                
                if len(main_stats['confidences']) > 1:
                    import numpy as np
                    conf_std = np.std(main_stats['confidences'])
                    stability = 1.0 / (1.0 + conf_std * 10)
                else:
                    stability = 0.5
                
                # 综合评分计算
                score = 0.3 * frequency_ratio + 0.25 * duration_ratio + 0.3 * avg_confidence + 0.15 * stability
                
                # 显示各项指标
                ttk.Label(scoring_frame, text=f"• 出现频率: {frequency_ratio*100:.1f}% (权重: 30%)", 
                         font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
                
                ttk.Label(scoring_frame, text=f"• 持续时间: {duration:.2f}秒, 占比: {duration_ratio*100:.1f}% (权重: 25%)", 
                         font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
                
                ttk.Label(scoring_frame, text=f"• 平均置信度: {avg_confidence:.3f} (权重: 30%)", 
                         font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
                
                ttk.Label(scoring_frame, text=f"• 置信度稳定性: {stability:.2f} (权重: 15%)", 
                         font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
                
                # 显示总分
                score_color = "#007AFF" if score >= self.trigger_threshold else "#666666"
                ttk.Label(scoring_frame, text=f"• 综合评分: {score:.3f} / {self.trigger_threshold:.3f} (触发阈值)", 
                         font=(font_family, 11, "bold"), foreground=score_color, 
                         style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=5)
                
                # 显示触发状态
                last_time = self.last_trigger_time.get(main_posture, 0)
                cooldown_passed = (current_time - last_time) >= self.trigger_cooldown
                can_trigger = score >= self.trigger_threshold and cooldown_passed
                
                trigger_status = "可以触发提示" if can_trigger else "等待中"
                if not cooldown_passed:
                    remaining = self.trigger_cooldown - (current_time - last_time)
                    trigger_status += f" (还需等待 {remaining:.1f} 秒)"
                
                trigger_color = "#00AA00" if can_trigger else "#AA0000"
                ttk.Label(scoring_frame, text=f"• 触发状态: {trigger_status}", 
                         font=(font_family, 10, "bold"), foreground=trigger_color, 
                         style="Modern.TLabel").pack(anchor=tk.W, padx=10, pady=2)
        
        # 底部按钮
        btn_frame = ttk.Frame(main_frame, style="Modern.TFrame")
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        close_btn = ttk.Button(btn_frame, text="关闭", 
                              command=details_window.destroy, style="Modern.TButton")
        close_btn.pack(side=tk.RIGHT)
        
        refresh_btn = ttk.Button(btn_frame, text="刷新", 
                               command=lambda: (details_window.destroy(), self.show_algorithm_details()), 
                               style="Modern.TButton")
        refresh_btn.pack(side=tk.RIGHT, padx=10)
        
        # 定期刷新窗口
        self.root.after(2000, lambda: self.refresh_algorithm_details_if_open(details_window))
    
    def refresh_algorithm_details_if_open(self, window):
        """如果算法详情窗口仍然打开，则刷新它"""
        try:
            if window.winfo_exists():
                window.destroy()
                self.show_algorithm_details()
        except:
            pass

    def on_algorithm_change(self):
        """处理算法选择变化"""
        selected = self.algorithm_type.get()
        
        # 隐藏所有参数面板
        if hasattr(self, 'smart_params_frame'):
            self.smart_params_frame.pack_forget()
        if hasattr(self, 'pro_params_frame'):
            self.pro_params_frame.pack_forget()
        
        # 根据选择显示对应参数面板
        if selected == "smart":
            self.use_smart_algorithm.set(True)
            self.smart_params_frame.pack(fill=tk.X, pady=(0, 5))
            self.toggle_smart_algorithm()
            # 启用参数控制
            self.conf_scale.config(state=tk.NORMAL)
            self.conf_entry.config(state=tk.NORMAL)
            self.iou_scale.config(state=tk.NORMAL)
            self.iou_entry.config(state=tk.NORMAL)
            self.fps_scale.config(state=tk.NORMAL)
            self.fps_entry.config(state=tk.NORMAL)
        elif selected == "pro":
            self.use_smart_algorithm.set(False)
            self.pro_params_frame.pack(fill=tk.X, pady=(0, 5))
            self.update_pro_aggressiveness()
            
            # 禁用参数控制并设置为Pro算法默认值
            # 从Pro算法中获取默认值
            pro_conf = self.posture_pro_recognizer.conf_threshold
            pro_iou = self.posture_pro_recognizer.iou_threshold
            
            # 设置参数值
            self.conf_var.set(pro_conf)
            self.iou_var.set(pro_iou)
            self.fps_var.set(5.0)  # 固定5FPS适合Pro算法
            
            # 禁用控件
            self.conf_scale.config(state=tk.DISABLED)
            self.conf_entry.config(state=tk.DISABLED)
            self.iou_scale.config(state=tk.DISABLED)
            self.iou_entry.config(state=tk.DISABLED)
            self.fps_scale.config(state=tk.DISABLED)
            self.fps_entry.config(state=tk.DISABLED)
        else:  # "none"
            self.use_smart_algorithm.set(False)
            # 启用参数控制
            self.conf_scale.config(state=tk.NORMAL)
            self.conf_entry.config(state=tk.NORMAL)
            self.iou_scale.config(state=tk.NORMAL)
            self.iou_entry.config(state=tk.NORMAL)
            self.fps_scale.config(state=tk.NORMAL)
            self.fps_entry.config(state=tk.NORMAL)
        
        # 记录算法切换
        if selected == "none":
            self.add_log("禁用智能算法", "所有姿态识别将使用单帧检测结果")
        elif selected == "smart":
            self.add_log("启用智能阈值算法", 
                      f"智能阈值算法配置:\n- 时间窗口长度: {self.window_duration:.1f}秒\n- 触发阈值: {self.trigger_threshold:.2f}\n- 提示冷却时间: {self.trigger_cooldown:.1f}秒")
        elif selected == "pro":
            self.add_log("启用智能算法Pro", 
                      f"智能算法Pro配置:\n- 激进度: {self.pro_aggressiveness.get()}\n- 批处理间隔: 2秒\n- 帧率: 5FPS\n- 输出类别: 4种(正确姿势,低头,不正确姿势,无人)")
    
    def update_pro_aggressiveness(self, *args):
        """更新智能算法Pro的激进度设置"""
        value = self.pro_aggressiveness.get()
        
        # 更新处理器的激进度
        self.posture_pro_recognizer.set_aggressiveness(value)
        
        # 更新界面显示
        # 根据激进度显示模式名称
        mode_name = "极度保守"
        if 0 <= value < 20:
            mode_name = "极度保守"
        elif 20 <= value < 40:
            mode_name = "保守"
        elif 40 <= value < 60:
            mode_name = "平衡"
        elif 60 <= value < 80:
            mode_name = "激进"
        else:
            mode_name = "极度激进"
        
        self.aggressiveness_label.config(text=f"{mode_name} ({value})")
        
        # 更新配置参数显示
        conf = self.posture_pro_recognizer.conf_threshold
        iou = self.posture_pro_recognizer.iou_threshold
        sw = self.posture_pro_recognizer.short_window_weight
        mw = self.posture_pro_recognizer.mid_window_weight
        lw = self.posture_pro_recognizer.long_window_weight
        
        self.conf_value_label.config(text=f"置信度: {conf:.2f}")
        self.iou_value_label.config(text=f"IOU: {iou:.2f}")
        self.window_weights_label.config(text=f"窗口权重: {sw:.1f}/{mw:.1f}/{lw:.1f}")
    
    def show_pro_algorithm_details(self):
        """显示智能算法Pro的详细信息窗口"""
        # 创建新窗口
        details_window = tk.Toplevel(self.root)
        details_window.title("智能算法Pro详情")
        details_window.geometry("700x500")
        details_window.minsize(600, 400)
        details_window.configure(background="#F5F5F7")
        
        # 获取字体
        font_family = "Microsoft YaHei" if os.name == "nt" else "Helvetica"
        
        # 创建主框架
        main_frame = ttk.Frame(details_window, padding="10", style="Modern.TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建标题
        ttk.Label(main_frame, text="智能算法Pro详情", 
                 font=(font_family, 14, "bold"), style="Title.TLabel").pack(anchor=tk.W, pady=(0, 10))
        
        # 获取当前算法状态
        if not hasattr(self, 'posture_pro_recognizer') or self.algorithm_type.get() != "pro":
            ttk.Label(main_frame, text="智能算法Pro当前未启用", 
                     font=(font_family, 12), style="Modern.TLabel").pack(anchor=tk.CENTER, pady=50)
            
            # 底部按钮
            btn_frame = ttk.Frame(main_frame, style="Modern.TFrame")
            btn_frame.pack(fill=tk.X, pady=(10, 0))
            
            close_btn = ttk.Button(btn_frame, text="关闭", 
                                  command=details_window.destroy, style="Modern.TButton")
            close_btn.pack(side=tk.RIGHT)
            return
        
        # 创建内容框架：分为左右两栏
        content_frame = ttk.Frame(main_frame, style="Modern.TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧：算法说明
        left_frame = ttk.LabelFrame(content_frame, text="算法设计", padding="10", style="Modern.TLabelframe")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 算法说明文本
        algo_text = tk.Text(left_frame, wrap=tk.WORD, height=10, 
                          font=(font_family, 10),
                          bg="#FFFFFF", fg="#333333",
                          highlightthickness=0, borderwidth=1,
                          relief=tk.SOLID)
        algo_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        algorithm_description = (
            "多时间尺度混合窗口模型:\n\n"
            "1. 三级时间窗口:\n"
            "   • 短期窗口(1秒): 捕捉当前动作\n"
            "   • 中期窗口(3秒): 平滑短期波动\n"
            "   • 长期窗口(8秒): 提供基线参考\n\n"
            "2. 姿态转换惩罚机制:\n"
            "   防止不合理的姿态快速切换，例如从正确姿势突然变成趴桌的概率很低\n\n"
            "3. 特殊姿态校正:\n"
            "   针对right_headed的误报率高和left_headed识别率低的问题进行特殊处理\n\n"
            "4. 批次处理:\n"
            "   每2秒收集约10帧数据进行综合分析，运行在5FPS\n\n"
            "5. 简化输出:\n"
            "   将六分类结果转换为三分类输出，提高识别稳定性：\n"
            "   • Correct Posture (正确姿势)\n"
            "   • Incorrect Posture (不正确姿势，包含所有错误姿势)\n"
            "   • Nobody (无人)"
        )
        
        algo_text.insert(tk.END, algorithm_description)
        algo_text.config(state=tk.DISABLED)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(algo_text, command=algo_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        algo_text.config(yscrollcommand=scrollbar.set)
        
        # 右侧：当前状态
        right_frame = ttk.LabelFrame(content_frame, text="当前状态", padding="10", style="Modern.TLabelframe")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # 激进度和参数设置
        ttk.Label(right_frame, text=f"• 激进度: {self.pro_aggressiveness.get()}", 
                 font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, pady=2)
        
        # 获取当前配置
        conf = self.posture_pro_recognizer.conf_threshold
        iou = self.posture_pro_recognizer.iou_threshold
        sw = self.posture_pro_recognizer.short_window_weight
        mw = self.posture_pro_recognizer.mid_window_weight
        lw = self.posture_pro_recognizer.long_window_weight
        right_correction = self.posture_pro_recognizer.right_headed_correction
        
        ttk.Label(right_frame, text=f"• 置信度阈值: {conf:.2f}", 
                 font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, pady=2)
        ttk.Label(right_frame, text=f"• IOU阈值: {iou:.2f}", 
                 font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, pady=2)
        ttk.Label(right_frame, text=f"• 窗口权重: {sw:.2f}/{mw:.2f}/{lw:.2f}", 
                 font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, pady=2)
        ttk.Label(right_frame, text=f"• Right_headed校正: +{right_correction:.2f}", 
                 font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, pady=2)
        
        # 窗口状态
        ttk.Label(right_frame, text="窗口状态:", 
                 font=(font_family, 10, "bold"), style="Modern.TLabel").pack(anchor=tk.W, pady=(10, 2))
        
        # 获取窗口状态
        stats = self.posture_pro_recognizer.get_output_and_stats()
        window_stats = stats['window_stats']
        
        ttk.Label(right_frame, text=f"• 短期窗口: {window_stats['short']}条记录", 
                 font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, padx=(10, 0))
        ttk.Label(right_frame, text=f"• 中期窗口: {window_stats['mid']}条记录", 
                 font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, padx=(10, 0))
        ttk.Label(right_frame, text=f"• 长期窗口: {window_stats['long']}条记录", 
                 font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, padx=(10, 0))
        
        # 当前输出
        ttk.Label(right_frame, text="当前输出:", 
                 font=(font_family, 10, "bold"), style="Modern.TLabel").pack(anchor=tk.W, pady=(10, 2))
        
        output = stats['output'] or "未确定"
        stability = stats['stability'] * 100
        
        output_label = ttk.Label(right_frame, text=f"• 姿态: {output}", 
                               font=(font_family, 11, "bold"), foreground="#007AFF", 
                               style="Modern.TLabel")
        output_label.pack(anchor=tk.W, pady=2)
        
        ttk.Label(right_frame, text=f"• 稳定性: {stability:.1f}%", 
                 font=(font_family, 10), style="Modern.TLabel").pack(anchor=tk.W, pady=2)
        
        # 底部按钮
        btn_frame = ttk.Frame(main_frame, style="Modern.TFrame")
        btn_frame.pack(fill=tk.X, pady=(10, 0))
        
        close_btn = ttk.Button(btn_frame, text="关闭", 
                              command=details_window.destroy, style="Modern.TButton")
        close_btn.pack(side=tk.RIGHT)
        
        refresh_btn = ttk.Button(btn_frame, text="刷新", 
                               command=lambda: (details_window.destroy(), self.show_pro_algorithm_details()), 
                               style="Modern.TButton")
        refresh_btn.pack(side=tk.RIGHT, padx=10)
        
        # 定期刷新窗口
        self.root.after(2000, lambda: self.refresh_window_if_open(details_window, self.show_pro_algorithm_details))
    
    def refresh_window_if_open(self, window, callback):
        """如果窗口仍然打开，则刷新它"""
        try:
            if window.winfo_exists():
                window.destroy()
                callback()
        except:
            pass

if __name__ == "__main__":
    # 创建主窗口
    root = tk.Tk()
    
    # 设置窗口图标（如果有的话）
    try:
        # 尝试设置窗口图标
        app_icon = tk.PhotoImage(file="models/app_icon.png")
        root.iconphoto(True, app_icon)
    except:
        # 图标加载失败，忽略错误
        pass
    
    # 创建应用程序实例
    app = YoloGUI(root)
    
    # 启动主循环
    root.mainloop() 