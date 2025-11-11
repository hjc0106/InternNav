#!/usr/bin/env python3
"""
InternVLA Real-world Inference Script
"""

# ========== 修复PIL线程安全问题 - 必须在最开头 ==========
import os
os.environ['MPLBACKEND'] = 'Agg'  # 使用非交互式后端
os.environ['DISPLAY'] = ''  # 禁用GUI显示

import sys
import glob
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import warnings

warnings.filterwarnings("ignore")

# Add project path
project_root = Path(".")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src/diffusion-policy'))

import torch.nn.functional as F

# from sageattention import sageattn
# F.scaled_dot_product_attention = sageattn

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent, Int8InternVLAN1AsyncAgent, Int4InternVLAN1AsyncAgent
# from internnav.agent.internvla_n1_agent_realworld_async_v2 import InternVLAN1AsyncAgent as InternVLAN1AsyncAgent_v2

class Args:
    def __init__(self):
        self.device = "cuda:0"
        self.model_path = "checkpoints/InternVLA-N1"
        self.resize_w = 384
        self.resize_h = 384
        self.num_history = 8
        self.camera_intrinsic = np.array([
            [386.5, 0.0, 328.9, 0.0],
            [0.0, 386.5, 244.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        self.plan_step_gap = 8
        # self.s2_dt = 8

args = Args()
print(f"Model path: {args.model_path}")
print(f"Device: {args.device}")
print(f"Image size: {args.resize_w}x{args.resize_h}")
print(f"History frames: {args.num_history}")

print("Loading model...")
agent = InternVLAN1AsyncAgent(args)

# Warm up model
print("Warming up model...")
dummy_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
dummy_depth = np.zeros((480, 640), dtype=np.float32)
dummy_pose = np.eye(4)
agent.reset()
agent.step(dummy_rgb, dummy_depth, dummy_pose, "hello", intrinsic=args.camera_intrinsic)
print("Model loaded successfully!")

# Configure data directory (single scene per folder)
scene_dir = 'assets/realworld_sample_data1'

# Check if instruction file exists
instruction_path = os.path.join(scene_dir, 'instruction.txt')
if not os.path.exists(instruction_path):
    print(f"Error: instruction.txt not found in {scene_dir}")
else:
    print(f"Scene directory: {scene_dir}")
    
    # Read instruction
    with open(instruction_path, 'r') as f:
        instruction = f.read().strip()
    print(f"Instruction: {instruction}")
    
    # Get all debug_raw images
    rgb_paths = sorted(glob.glob(os.path.join(scene_dir, 'debug_raw_*.jpg')))
    print(f"\nFound {len(rgb_paths)} images")
    # Show first few image names
    print("\nFirst 5 images:")
    for i, path in enumerate(rgb_paths[:5]):
        print(f"  {i+1}. {os.path.basename(path)}")


from utils import annotate_image

# Reset agent
agent.reset()
print(f"{'='*80}")
print(f"Processing scene: {os.path.basename(scene_dir)}")
print(f"Instruction: '{instruction}'")
print(f"Total images: {len(rgb_paths)}")
print(f"{'='*80}\n")

action_seq = []
look_down = False

save_dir = 'test_data/'
os.makedirs(save_dir, exist_ok=True)
# Process each image
for i, rgb_path in enumerate(rgb_paths):
    # Check if this is a look_down image
    look_down = ('look_down' in rgb_path)
    
    # Extract image ID from filename (e.g., debug_raw_0003.jpg -> 0003)
    basename = os.path.basename(rgb_path)
    if look_down:
        # e.g., debug_raw_0010_look_down.jpg -> 0010
        image_id = basename.replace('debug_raw_', '').replace('_look_down.jpg', '')
    else:
        # e.g., debug_raw_0003.jpg -> 0003
        image_id = basename.replace('debug_raw_', '').replace('.jpg', '')
        
    # Read RGB image
    rgb = np.asarray(Image.open(rgb_path).convert('RGB'))
    
    # Create dummy depth image (not available in test data)
    # !Note You must full in depth to model
    depth = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.float32)
    
    # Create dummy camera pose
    camera_pose = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Run model or just save image
    # print(f"[{i+1}/{len(rgb_paths)}] Running model inference: {os.path.basename(rgb_path)}")
    dual_sys_output = agent.step(
        rgb, 
        depth, 
        camera_pose, 
        instruction, 
        intrinsic=args.camera_intrinsic,
        look_down=look_down
    )
    
    # Print output results
    if dual_sys_output.output_action is not None and dual_sys_output.output_action != []:
        print(f"  Output action: {dual_sys_output.output_action}")
        # action_seq.extend(s2_output.output_action)
    else:
        print(f"output_trajectory: {dual_sys_output.output_trajectory.tolist()[-1]}")
        if dual_sys_output.output_pixel is not None:
            print(f"output_pixel: {dual_sys_output.output_pixel}")
            annotate_image(image_id, rgb, 'traj', dual_sys_output.output_trajectory.tolist(), dual_sys_output.output_pixel, save_dir)


print(f"\nScene {os.path.basename(scene_dir)} completed!")