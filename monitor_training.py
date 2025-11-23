#!/usr/bin/env python3
"""
监控训练进度的脚本
"""
import os
import json
import time
from datetime import datetime

def monitor_training_progress(output_dir):
    """监控训练进度"""
    print(f"开始监控训练进度，输出目录: {output_dir}")
    
    last_epoch = 0
    while True:
        # 检查是否有训练历史文件
        training_history_file = os.path.join(output_dir, "training_history.json")
        if os.path.exists(training_history_file):
            try:
                with open(training_history_file, 'r') as f:
                    history = json.load(f)
                
                epochs_completed = history.get("epochs_completed", 0)
                if epochs_completed > last_epoch:
                    last_epoch = epochs_completed
                    train_losses = history.get("train_losses", [])
                    if train_losses:
                        current_loss = train_losses[-1]
                        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 训练轮数: {epochs_completed}, 当前损失: {current_loss:.6f}")
                
                # 检查是否有验证结果
                validation_file = os.path.join(output_dir, "validation_results.json")
                if os.path.exists(validation_file):
                    with open(validation_file, 'r') as f:
                        validation = json.load(f)
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 训练完成!")
                    print(f"测试损失: {validation.get('test_loss', 'N/A')}")
                    print(f"物理损失: {validation.get('physics_loss', 'N/A')}")
                    break
                    
            except Exception as e:
                print(f"读取训练历史文件时出错: {e}")
        
        # 检查是否有最终模型文件
        final_model_file = os.path.join(output_dir, "final_model.pth")
        if os.path.exists(final_model_file):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 检测到最终模型文件，训练可能已完成")
            break
            
        time.sleep(30)  # 每30秒检查一次

if __name__ == "__main__":
    output_dir = "results_long_run"
    monitor_training_progress(output_dir)