#!/usr/bin/env python3
"""
监控训练进度的脚本
"""

import os
import json
import time
import argparse
from datetime import datetime

def monitor_training(output_dir, check_interval=30):
    """
    监控训练进度
    
    Args:
        output_dir: 输出目录
        check_interval: 检查间隔（秒）
    """
    print(f"开始监控训练进度，输出目录: {output_dir}")
    print(f"检查间隔: {check_interval}秒")
    
    last_epoch = -1
    last_loss = float('inf')
    
    while True:
        try:
            # 检查训练历史文件
            history_file = os.path.join(output_dir, 'training_history.json')
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                
                # 获取当前训练状态
                current_epoch = history.get('epochs_completed', 0)
                train_losses = history.get('train_losses', [])
                
                if train_losses:
                    current_loss = train_losses[-1]
                else:
                    current_loss = float('inf')
                
                # 如果有更新，打印信息
                if current_epoch > last_epoch or abs(current_loss - last_loss) > 1e-6:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] 训练进度更新:")
                    print(f"  当前轮次: {current_epoch}")
                    print(f"  当前损失: {current_loss:.8f}")
                    
                    if train_losses and len(train_losses) > 1:
                        prev_loss = train_losses[-2] if len(train_losses) > 1 else train_losses[-1]
                        loss_change = ((current_loss - prev_loss) / prev_loss) * 100 if prev_loss != 0 else 0
                        print(f"  损失变化: {loss_change:+.4f}%")
                    
                    last_epoch = current_epoch
                    last_loss = current_loss
            
            # 检查验证结果文件
            validation_file = os.path.join(output_dir, 'validation_results.json')
            if os.path.exists(validation_file):
                with open(validation_file, 'r') as f:
                    validation = json.load(f)
                
                test_loss = validation.get('test_loss', 0)
                physics_loss = validation.get('physics_loss', 0)
                timestamp = validation.get('timestamp', '')
                
                print(f"[{timestamp}] 验证结果:")
                print(f"  测试损失: {test_loss:.8f}")
                print(f"  物理损失: {physics_loss:.8f}")
            
            # 检查是否有最终模型
            final_model_file = os.path.join(output_dir, 'final_model.pth')
            if os.path.exists(final_model_file):
                print("训练完成！最终模型已生成。")
                break
                
        except Exception as e:
            print(f"监控出错: {str(e)}")
        
        # 等待下一次检查
        time.sleep(check_interval)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='监控训练进度')
    parser.add_argument('--output_dir', type=str, default='results_long_run',
                        help='输出目录')
    parser.add_argument('--interval', type=int, default=30,
                        help='检查间隔（秒）')
    
    args = parser.parse_args()
    monitor_training(args.output_dir, args.interval)