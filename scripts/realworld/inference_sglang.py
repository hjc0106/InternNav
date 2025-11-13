#!/usr/bin/env python3
"""
InternVLA Real-world Inference with SGLang Optimizations
使用 SGLang 风格优化的推理脚本
"""

# ========== 修复PIL线程安全问题 - 必须在最开头 ==========
import os
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''

import sys
import glob
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import warnings
import time

warnings.filterwarnings("ignore")

# Add project path
project_root = Path(".")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src/diffusion-policy'))

# 导入 SGLang 优化的 Agent
from internnav.agent.internvla_n1_agent_sglang import Int8InternVLAN1SGLangAgent


class Args:
    """配置参数"""
    def __init__(self):
        self.device = "cuda:0"
        self.model_path = "checkpoints/InternVLA-N1"
        
        # 图像配置
        self.resize_w = 384
        self.resize_h = 384
        
        # 历史配置
        self.num_history = 8  # 减少历史帧数以加速
        
        # 相机内参
        self.camera_intrinsic = np.array([
            [386.5, 0.0, 328.9, 0.0],
            [0.0, 386.5, 244.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # 推理配置
        self.plan_step_gap = 8
        self.max_new_tokens = 64  # 减少生成长度以加速
        
        # ============ SGLang 配置 ============
        self.sglang_max_cache = 2048  # KV Cache 最大长度
        self.sglang_reset_interval = 15  # 每15步重置缓存
        self.sglang_enable_metrics = True  # 启用性能监控
        self.attn_implementation = "flash_attention_2"
        # ====================================


def load_scene_data(scene_dir):
    """加载场景数据"""
    # 读取指令
    instruction_path = os.path.join(scene_dir, 'instruction.txt')
    if not os.path.exists(instruction_path):
        raise FileNotFoundError(f"instruction.txt not found in {scene_dir}")
    
    with open(instruction_path, 'r') as f:
        instruction = f.read().strip()
    
    # 获取所有图像
    rgb_paths = sorted(glob.glob(os.path.join(scene_dir, 'debug_raw_*.jpg')))
    
    return instruction, rgb_paths


def main():
    """主函数"""
    print(f"\n{'='*80}")
    print(f"InternVLA-N1 Inference with SGLang Optimizations")
    print(f"{'='*80}\n")
    
    # 创建配置
    args = Args()
    
    print("Configuration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Device: {args.device}")
    print(f"  Image size: {args.resize_w}x{args.resize_h}")
    print(f"  History frames: {args.num_history}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    print(f"\n[SGLang] Configuration:")
    print(f"  Max cache length: {args.sglang_max_cache}")
    print(f"  Reset interval: {args.sglang_reset_interval}")
    print(f"  Attention implementation: {args.attn_implementation}")
    print()
    
    # 加载模型
    print("Loading model with SGLang optimizations...")
    start_time = time.time()
    agent = Int8InternVLAN1SGLangAgent(args)
    print(f"✅ Model loaded in {time.time() - start_time:.2f}s\n")
    
    # 预热模型
    print("Warming up model...")
    dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_depth = np.zeros((480, 640), dtype=np.float32)
    dummy_pose = np.eye(4)
    agent.reset()
    agent.step(dummy_rgb, dummy_depth, dummy_pose, "hello", intrinsic=args.camera_intrinsic)
    print("✅ Model warmed up\n")
    
    # 配置数据目录
    scene_dir = 'assets/realworld_sample_data1'
    
    # 加载场景数据
    print(f"Loading scene: {scene_dir}")
    instruction, rgb_paths = load_scene_data(scene_dir)
    
    print(f"\nScene Information:")
    print(f"  Directory: {scene_dir}")
    print(f"  Instruction: '{instruction}'")
    print(f"  Total images: {len(rgb_paths)}")
    print(f"\nFirst 5 images:")
    for i, path in enumerate(rgb_paths[:5]):
        print(f"  {i+1}. {os.path.basename(path)}")
    
    # 导入可视化工具
    sys.path.insert(0, str(project_root / 'scripts/realworld'))
    from utils import annotate_image
    
    # 重置 agent
    agent.reset()
    print(f"\n{'='*80}")
    print(f"Starting inference...")
    print(f"{'='*80}\n")
    
    # 创建保存目录
    save_dir = 'test_data_sglang/'
    os.makedirs(save_dir, exist_ok=True)
    
    # 推理统计
    total_inference_time = 0
    total_frames = 0
    
    # 处理每一帧
    for i, rgb_path in enumerate(rgb_paths):
        # 检查是否为 look_down 图像
        look_down = ('look_down' in rgb_path)
        
        # 提取图像 ID
        basename = os.path.basename(rgb_path)
        if look_down:
            image_id = basename.replace('debug_raw_', '').replace('_look_down.jpg', '')
        else:
            image_id = basename.replace('debug_raw_', '').replace('.jpg', '')
        
        # 读取图像
        rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
        depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
        camera_pose = np.eye(4)
        
        # 运行推理
        frame_start = time.time()
        dual_sys_output = agent.step(
            rgb,
            depth,
            camera_pose,
            instruction,
            intrinsic=args.camera_intrinsic,
            look_down=look_down
        )
        frame_time = time.time() - frame_start
        
        total_inference_time += frame_time
        total_frames += 1
        
        # 打印结果
        if dual_sys_output.output_action is not None and dual_sys_output.output_action != []:
            print(f"  Output action: {dual_sys_output.output_action}")
        else:
            if dual_sys_output.output_trajectory is not None:
                print(f"output_trajectory: {dual_sys_output.output_trajectory.tolist()[-1]}")
                if dual_sys_output.output_pixel is not None:
                    print(f"output_pixel: {dual_sys_output.output_pixel}")
                    annotate_image(
                        image_id, 
                        rgb, 
                        'traj', 
                        dual_sys_output.output_trajectory.tolist(), 
                        dual_sys_output.output_pixel, 
                        save_dir
                    )
    
    # 打印统计信息
    print(f"\n{'='*80}")
    print(f"Inference completed!")
    print(f"{'='*80}")
    print(f"Statistics:")
    print(f"  Total frames: {total_frames}")
    print(f"  Total time: {total_inference_time:.2f}s")
    print(f"  Average time per frame: {total_inference_time/total_frames:.2f}s")
    print(f"  FPS: {total_frames/total_inference_time:.2f}")
    
    # 打印 SGLang 性能指标
    agent.print_metrics()
    
    print(f"\n✅ Results saved to: {save_dir}")
    print(f"✅ Scene {os.path.basename(scene_dir)} completed!\n")


if __name__ == "__main__":
    main()

