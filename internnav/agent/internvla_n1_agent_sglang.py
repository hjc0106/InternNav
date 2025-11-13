"""
InternVLA-N1 Agent with SGLang-style optimizations
融合 SGLang 的缓存管理策略到 InternVLA-N1
"""

import copy
import itertools
import os
import re
import time
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from collections import OrderedDict
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import S2Output, split_and_clean, traj_to_actions

DEFAULT_IMAGE_TOKEN = "<image>"


class SGLangCacheManager:
    """
    SGLang 风格的 KV Cache 管理器
    实现 RadixAttention 的核心思想：智能缓存管理
    """
    
    def __init__(
        self,
        max_cache_length: int = 2048,
        reset_interval: int = 15,
        enable_metrics: bool = True
    ):
        self.max_cache_length = max_cache_length
        self.reset_interval = reset_interval
        self.enable_metrics = enable_metrics
        
        self.step_counter = 0
        self.total_tokens_saved = 0
        self.total_resets = 0
        self.total_trims = 0
        
    def should_reset(self) -> bool:
        """判断是否需要重置缓存（类似 SGLang 的会话管理）"""
        return self.step_counter > 0 and self.step_counter % self.reset_interval == 0
    
    def manage_cache(self, past_key_values):
        """
        管理 KV Cache（类似 SGLang 的 PagedAttention）
        
        策略：
        1. 定期完全重置
        2. 超长时截断到最近的 tokens
        """
        if past_key_values is None:
            return None
        
        # 策略 1: 定期重置
        if self.should_reset():
            self.total_resets += 1
            cache_length = past_key_values[0][0].shape[2]
            if self.enable_metrics:
                print(f"🔄 [SGLang] Reset cache at step {self.step_counter} (cleared {cache_length} tokens)")
            torch.cuda.empty_cache()
            return None
        
        # 策略 2: 长度限制
        cache_length = past_key_values[0][0].shape[2]
        if cache_length > self.max_cache_length:
            self.total_trims += 1
            tokens_removed = cache_length - self.max_cache_length
            self.total_tokens_saved += tokens_removed
            
            new_past_key_values = []
            for layer_past in past_key_values:
                new_layer = tuple(
                    kv[:, :, -self.max_cache_length:, :] 
                    for kv in layer_past
                )
                new_past_key_values.append(new_layer)
            
            if self.enable_metrics:
                print(f"✂️ [SGLang] Trimmed cache from {cache_length} to {self.max_cache_length}")
            
            return tuple(new_past_key_values)
        
        return past_key_values
    
    def increment_step(self):
        """增加步数计数"""
        self.step_counter += 1
    
    def reset_counter(self):
        """重置计数器"""
        self.step_counter = 0
    
    def get_metrics(self) -> dict:
        """获取性能指标"""
        return {
            'total_steps': self.step_counter,
            'total_resets': self.total_resets,
            'total_trims': self.total_trims,
            'tokens_saved': self.total_tokens_saved,
        }


class Int8InternVLAN1SGLangAgent:
    """
    Int8 量化 + SGLang 优化的 InternVLA-N1 Agent
    结合了量化和智能缓存管理的优势
    """
    
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"args.model_path: {args.model_path}")
        
        # 8-bit 量化配置
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            llm_int8_enable_fp32_cpu_offload=False,
            llm_int8_skip_modules=["navdp"]
        )
        
        # 加载模型
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=bnb_config,
            torch_dtype="auto",
            # torch_dtype=torch.bfloat16,
            attn_implementation=args.attn_implementation,  # 默认使用 sdpa
            device_map={"": torch.device("cuda:0")},
        )
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        self.processor.tokenizer.padding_side = 'left'
        
        # 配置参数
        self.resize_w = args.resize_w
        self.resize_h = args.resize_h
        self.num_history = args.num_history
        self.PLAN_STEP_GAP = args.plan_step_gap
        self.max_new_tokens = getattr(args, 'max_new_tokens', 128)
        
        # SGLang 缓存管理器
        self.cache_manager = SGLangCacheManager(
            max_cache_length=getattr(args, 'sglang_max_cache', 2048),
            reset_interval=getattr(args, 'sglang_reset_interval', 15),
            enable_metrics=getattr(args, 'sglang_enable_metrics', True)
        )
        
        # 提示模板
        prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. Where should you go next to stay on track? Please output the next waypoint's coordinates in the image. Please output STOP when you have successfully completed the task."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        
        self.conjunctions = [
            'you can see ',
            'in front of you is ',
            'there is ',
            'you can spot ',
            'you are toward the ',
            'ahead of you is ',
            'in your sight is ',
        ]
        
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3],
            "↓": [5],
        })
        
        # 状态变量
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None
        self.last_s2_idx = -100
        
        # 输出缓存
        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        
        print(f"✅ [SGLang] Agent initialized with cache management")
        print(f"   - Max cache length: {self.cache_manager.max_cache_length}")
        print(f"   - Reset interval: {self.cache_manager.reset_interval}")
        print(f"   - Attention: {getattr(args, 'attn_implementation', 'sdpa')}")
    
    def reset(self):
        """重置 Agent 状态"""
        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None
        
        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None
        
        self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 重置缓存管理器
        self.cache_manager.reset_counter()
        
        # 清理 GPU 缓存
        torch.cuda.empty_cache()
        print(f"🔄 [SGLang] Agent reset, GPU cache cleared")
    
    def parse_actions(self, output):
        """解析动作序列"""
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)
    
    def step_no_infer(self, rgb, depth, pose):
        """不进行推理，只保存历史"""
        image = Image.fromarray(rgb).convert('RGB')
        image = image.resize((self.resize_w, self.resize_h))
        self.rgb_list.append(image)
        image.save(f"{self.save_dir}/debug_raw_{self.episode_idx:04d}.jpg")
        self.episode_idx += 1
        
    def step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        """执行一步推理"""
        dual_sys_output = S2Output()
        no_output_flag = self.output_action is None and self.output_latent is None
        
        if (self.episode_idx - self.last_s2_idx > self.PLAN_STEP_GAP) or look_down or no_output_flag:
            self.output_action, self.output_latent, self.output_pixel = self.step_s2(
                rgb, depth, pose, instruction, intrinsic, look_down
            )
            self.last_s2_idx = self.episode_idx
            dual_sys_output.output_pixel = self.output_pixel
            self.pixel_goal_rgb = copy.deepcopy(rgb)
            self.pixel_goal_depth = copy.deepcopy(depth)
        else:
            self.step_no_infer(rgb, depth, pose)
        
        if self.output_action is not None:
            dual_sys_output.output_action = copy.deepcopy(self.output_action)
            self.output_action = None
        elif self.output_latent is not None:
            processed_pixel_rgb = np.array(Image.fromarray(self.pixel_goal_rgb).resize((224, 224))) / 255
            processed_pixel_depth = np.array(Image.fromarray(self.pixel_goal_depth).resize((224, 224)))
            processed_rgb = np.array(Image.fromarray(rgb).resize((224, 224))) / 255
            processed_depth = np.array(Image.fromarray(depth).resize((224, 224)))
            
            rgbs = (
                torch.stack([torch.from_numpy(processed_pixel_rgb), torch.from_numpy(processed_rgb)])
                .unsqueeze(0)
                .to(self.device)
            )
            depths = (
                torch.stack([torch.from_numpy(processed_pixel_depth), torch.from_numpy(processed_depth)])
                .unsqueeze(0)
                .unsqueeze(-1)
                .to(self.device)
            )
            trajectories = self.step_s1(self.output_latent, rgbs, depths)
            dual_sys_output.output_trajectory = traj_to_actions(trajectories, use_discrate_action=False)
        
        return dual_sys_output
    
    def step_s2(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        """System 2: VLM 推理"""
        image = Image.fromarray(rgb).convert('RGB')
        
        if not look_down:
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
            image.save(f"{self.save_dir}/debug_raw_{self.episode_idx:04d}.jpg")
        else:
            image.save(f"{self.save_dir}/debug_raw_{self.episode_idx:04d}_look_down.jpg")
        
        if not look_down:
            self.conversation_history = []
            self.past_key_values = None
            
            sources = copy.deepcopy(self.conversation)
            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', instruction)
            cur_images = self.rgb_list[-1:]
            
            if self.episode_idx == 0:
                history_id = []
            else:
                history_id = np.unique(np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)).tolist()
                placeholder = (DEFAULT_IMAGE_TOKEN + '\n') * len(history_id)
                sources[0]["value"] += f' These are your historical observations: {placeholder}.'
            
            history_id = sorted(history_id)
            self.input_images = [self.rgb_list[i] for i in history_id] + cur_images
            input_img_id = 0
            self.episode_idx += 1
        else:
            self.input_images.append(image)
            input_img_id = -1
            assert self.llm_output != "", "Last llm_output should not be empty when look down"
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
            self.conversation_history.append(
                {'role': 'assistant', 'content': [{'type': 'text', 'text': self.llm_output}]}
            )
        
        prompt = self.conjunctions[0] + DEFAULT_IMAGE_TOKEN
        sources[0]["value"] += f" {prompt}."
        prompt_instruction = copy.deepcopy(sources[0]["value"])
        parts = split_and_clean(prompt_instruction)
        
        content = []
        for i in range(len(parts)):
            if parts[i] == "<image>":
                content.append({"type": "image", "image": self.input_images[input_img_id]})
                input_img_id += 1
            else:
                content.append({"type": "text", "text": parts[i]})
        
        self.conversation_history.append({'role': 'user', 'content': content})
        
        text = self.processor.apply_chat_template(
            self.conversation_history, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt").to(self.device)
        
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                past_key_values=self.past_key_values,
                return_dict_in_generate=True,
                raw_input_ids=copy.deepcopy(inputs.input_ids),
            )
        output_ids = outputs.sequences
        
        t1 = time.time()
        self.llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        
        with open(f"{self.save_dir}/llm_output_{self.episode_idx:04d}.txt", 'w') as f:
            f.write(self.llm_output)
        
        self.last_output_ids = copy.deepcopy(output_ids[0])
        self.past_key_values = copy.deepcopy(outputs.past_key_values)
        
        # ============ SGLang 风格的缓存管理 ============
        self.past_key_values = self.cache_manager.manage_cache(self.past_key_values)
        self.cache_manager.increment_step()
        
        # 监控信息
        if self.past_key_values is not None:
            cache_len = self.past_key_values[0][0].shape[2]
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            print(f"📊 [SGLang] Step {self.cache_manager.step_counter}: "
                  f"cache={cache_len}, GPU={gpu_mem:.2f}GB, time={t1-t0:.2f}s")
        else:
            print(f"📊 [SGLang] Step {self.cache_manager.step_counter}: "
                  f"cache=None (reset), time={t1-t0:.2f}s")
        # ============ 结束 SGLang 管理 ============
        
        print(f"output {self.episode_idx}  {self.llm_output} cost: {t1 - t0}s")
        
        if bool(re.search(r'\d', self.llm_output)):
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            pixel_goal = [int(coord[1]), int(coord[0])]
            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
            pixel_values = inputs.pixel_values
            
            t0 = time.time()
            with torch.no_grad():
                traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)
                print(f'generate latents cost: {time.time() - t0}s')
                return None, traj_latents, pixel_goal
        else:
            action_seq = self.parse_actions(self.llm_output)
            return action_seq, None, None
    
    def step_s1(self, latent, rgb, depth):
        """System 1: NavDP 推理"""
        all_trajs = self.model.generate_traj(latent, rgb, depth, use_async=True)
        return all_trajs
    
    def print_metrics(self):
        """打印性能指标"""
        metrics = self.cache_manager.get_metrics()
        print(f"\n{'='*60}")
        print(f"[SGLang] Performance Metrics:")
        print(f"  Total steps: {metrics['total_steps']}")
        print(f"  Cache resets: {metrics['total_resets']}")
        print(f"  Cache trims: {metrics['total_trims']}")
        print(f"  Tokens saved: {metrics['tokens_saved']}")
        print(f"{'='*60}\n")

