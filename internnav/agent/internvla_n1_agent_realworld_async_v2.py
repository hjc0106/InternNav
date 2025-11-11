import copy
import itertools
import os
import re
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent.parent))

from collections import OrderedDict

from PIL import Image
from transformers import AutoProcessor

from internnav.model.basemodel.internvla_n1.internvla_n1 import InternVLAN1ForCausalLM
from internnav.model.utils.vln_utils import S2Output, split_and_clean, traj_to_actions
from transformers import BitsAndBytesConfig

DEFAULT_IMAGE_TOKEN = "<image>"


class InternVLAN1AsyncAgent:
    """
    异步版本的InternVLA导航Agent - V2
    
    优化策略：提前执行S2
    - 在使用历史latent执行S1期间，异步预执行下一轮S2
    - 当需要新latent时，S2已经完成，直接使用
    
    时间线示例：
    T1: S2(4s) + S1(0.2s) → 启动异步S2预执行
    T2: S1(0.2s)  | 异步S2(0-0.2s)
    T3: S1(0.2s)  | 异步S2(0.2-0.4s)
    ...
    T8: S1(0.2s)  | 异步S2(1.2-1.4s)
    [等待异步S2完成，剩余2.6s]
    T9: 使用新latent + S1(0.2s)
    
    性能提升：1.4s (原本需要在T9等待4s，现在只需等2.6s)
    """
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"args.model_path{args.model_path}")
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": torch.device("cuda:0")},
        )
        self.model.eval()
        self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(args.model_path)
        self.processor.tokenizer.padding_side = 'left'

        self.resize_w = args.resize_w
        self.resize_h = args.resize_h
        self.num_history = args.num_history
        self.PLAN_STEP_GAP = args.plan_step_gap

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

        self.actions2idx = OrderedDict(
            {
                'STOP': [0],
                "↑": [1],
                "←": [2],
                "→": [3],
                "↓": [5],
            }
        )

        self.rgb_list = []
        self.depth_list = []
        self.pose_list = []
        self.episode_idx = 0
        self.conversation_history = []
        self.llm_output = ""
        self.past_key_values = None
        self.last_s2_idx = -100

        # output
        self.output_action = None
        self.output_latent = None
        self.output_pixel = None
        self.pixel_goal_rgb = None
        self.pixel_goal_depth = None

        # ========== 异步S2预执行相关 ==========
        self.s2_dt = args.s2_dt
        self.s2_prefetch_thread = None
        self.s2_prefetch_lock = threading.Lock()
        self.s2_prefetch_result = None  # 存储预执行的S2结果
        self.s2_prefetch_ready = False
        self.s2_prefetch_running = False
        self.s2_thread_stop_flag = False
        
        # 用于传递给异步S2的参数
        self.s2_pending_args = None
        
        print("异步S2预执行Agent初始化完成")

    def _s2_prefetch_worker(self, rgb, depth, pose, instruction, intrinsic, look_down):
        """
        后台线程：提前执行S2
        """
        try:
            print(f"[异步S2] 开始预执行 (episode {self.episode_idx})")
            t0 = time.time()
            
            # 执行S2
            output_action, output_latent, output_pixel = self.step_s2(
                rgb, depth, pose, instruction, intrinsic, look_down
            )
            
            t1 = time.time()
            print(f"[异步S2] 预执行完成，耗时 {t1-t0:.3f}s")
            
            # 保存结果
            with self.s2_prefetch_lock:
                self.s2_prefetch_result = (output_action, output_latent, output_pixel)
                self.s2_prefetch_ready = True
                self.s2_prefetch_running = False
                
        except Exception as e:
            print(f"[异步S2] 错误: {e}")
            import traceback
            traceback.print_exc()
            with self.s2_prefetch_lock:
                self.s2_prefetch_running = False

    def _start_s2_prefetch(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        """
        启动异步S2预执行
        """
        with self.s2_prefetch_lock:
            if self.s2_prefetch_running:
                print("[异步S2] 警告: 上一次预执行还在进行中，跳过")
                return
            
            self.s2_prefetch_running = True
            self.s2_prefetch_ready = False
            self.s2_prefetch_result = None
        
        # 启动新线程
        self.s2_prefetch_thread = threading.Thread(
            target=self._s2_prefetch_worker,
            args=(rgb, depth, pose, instruction, intrinsic, look_down),
            daemon=True
        )
        self.s2_prefetch_thread.start()
        print(f"[异步S2] 已启动预执行线程")

    def _get_s2_prefetch_result(self, wait_timeout=1.0):
        """
        获取异步S2的结果
        如果还未完成，会等待
        """
        if not self.s2_prefetch_ready:
            print(f"[异步S2] 结果未就绪，等待完成...")
            start_time = time.time()
            
            # 等待结果就绪
            while not self.s2_prefetch_ready:
                time.sleep(0.1)
                if time.time() - start_time > wait_timeout:
                    print(f"[异步S2] 警告: 等待超时")
                    return None, None, None
        
        # 取出结果
        with self.s2_prefetch_lock:
            result = self.s2_prefetch_result
            self.s2_prefetch_result = None
            self.s2_prefetch_ready = False
        
        print(f"[异步S2] 获取预执行结果成功")
        return result if result else (None, None, None)

    def reset(self):
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
        
        # 重置异步状态
        with self.s2_prefetch_lock:
            self.s2_prefetch_result = None
            self.s2_prefetch_ready = False

        self.save_dir = "test_data/" + datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.save_dir, exist_ok=True)

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)

    def step_no_infer(self, rgb, depth, pose):
        image = Image.fromarray(rgb).convert('RGB')
        image = image.resize((self.resize_w, self.resize_h))
        self.rgb_list.append(image)
        # image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}.jpg")
        self.episode_idx += 1   

    def trajectory_tovw(self, trajectory, kp=1.0):
        subgoal = trajectory[-1]
        linear_vel, angular_vel = kp * np.linalg.norm(subgoal[:2]), kp * subgoal[2]
        linear_vel = np.clip(linear_vel, 0, 0.5)
        angular_vel = np.clip(angular_vel, -0.5, 0.5)
        return linear_vel, angular_vel

    def step(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        """
        主step函数（异步S2预执行版本）
        
        执行逻辑：
        1. 判断是否需要执行S2
        2. 如果需要：
           - 优先检查是否有预执行的S2结果
           - 如果没有，同步执行S2
           - 执行S1
           - 启动下一轮S2预执行
        3. 如果不需要：
           - 使用历史latent执行S1
           - 如果还没启动S2预执行，且距离下次S2还有足够时间，启动预执行
        """
        dual_sys_output = S2Output()
        no_output_flag = self.output_action is None and self.output_latent is None
        
        # ========== 情况1: 需要执行S2 ==========
        if (self.episode_idx - self.last_s2_idx >= self.PLAN_STEP_GAP) or look_down or no_output_flag:
            print(f"\n[Step] Episode {self.episode_idx}: 需要S2")
            
            # 检查是否有预执行在运行或已完成
            if self.s2_prefetch_running or self.s2_prefetch_ready:
                # 有预执行，等待完成（如果已完成则立即返回）
                if self.s2_prefetch_ready:
                    print(f"[Step] 预执行已完成，直接使用")
                else:
                    while not self.s2_prefetch_ready:
                        time.sleep(0.1)
                        print(f"[Step] 预执行进行中，等待完成...")
                
                self.output_action, self.output_latent, self.output_pixel = self._get_s2_prefetch_result()
            
                if self.output_latent is not None or self.output_action is not None:
                    print(f"[Step] 获取预执行结果成功")
                else:
                    # 预执行失败或超时，回退到同步执行
                    raise RuntimeError("预执行结果无效")
            else:
                # 没有预执行任务（首次调用），同步执行
                print(f"[Step] 无预执行任务，同步执行S2")
                t0 = time.time()
                self.output_action, self.output_latent, self.output_pixel = self.step_s2(
                    rgb, depth, pose, instruction, intrinsic, look_down
                )
                print(f"[Step] 同步S2耗时: {time.time()-t0:.3f}s")
            
            self.last_s2_idx = self.episode_idx
            dual_sys_output.output_pixel = self.output_pixel
            self.pixel_goal_rgb = copy.deepcopy(rgb)
            self.pixel_goal_depth = copy.deepcopy(depth)
            
            # 如果获得latent，启动下一轮S2预执行
            if self.output_latent is not None:
                # 预测下一轮需要S2的时刻，提前启动
                next_s2_episode = self.episode_idx + self.PLAN_STEP_GAP
                print(f"[Step] 预计在 Episode {next_s2_episode} 需要下一轮S2，现在启动预执行")
                
                # 启动异步S2（使用当前的rgb/depth/pose作为下一轮的输入）
                # 注意：这里使用当前帧，实际应该等到下一轮再用最新的
                # 为了简化，我们在下面的"不需要S2"分支中启动
                pass
        else:
            # ========== 情况2: 不需要执行S2，使用历史latent ==========
            self.step_no_infer(rgb, depth, pose)
            steps_until_next_s2 = self.PLAN_STEP_GAP - (self.episode_idx - self.last_s2_idx)
            print(f"[Step] Episode {self.episode_idx}: 使用历史latent (距下次S2还有{steps_until_next_s2}步)")            
            # 判断是否应该启动S2预执行
            # 策略：在距离下次S2还有足够步数时启动（例如还有PLAN_STEP_GAP-1步时）
            should_prefetch = (
                steps_until_next_s2 == self.PLAN_STEP_GAP - self.s2_dt and  # 下一步就需要S2
                not self.s2_prefetch_running and  # 没有正在运行的预执行
                not self.s2_prefetch_ready and  # 没有就绪的结果
                self.output_latent is not None  # 当前使用的是latent模式
            )
            
            if should_prefetch:
                print(f"[Step] 启动S2预执行（提前准备下一轮）")
                self._start_s2_prefetch(rgb, depth, pose, instruction, intrinsic, True)

        # ========== 返回当前可用的输出 ==========
        if self.output_action is not None:
            # 离散动作模式
            dual_sys_output.output_action = copy.deepcopy(self.output_action)
            self.output_action = None
            print(f"[Step] 返回离散动作")
            
        elif self.output_latent is not None:
            # 轨迹模式：执行S1
            print(f"[Step] 执行S1生成轨迹")
            t0 = time.time()
            
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
            
            print(f"[Step] S1耗时: {time.time()-t0:.3f}s")

        return dual_sys_output

    def step_s2(self, rgb, depth, pose, instruction, intrinsic, look_down=False):
        """
        Stage 2: VLA推理，生成latent或离散动作
        """
        image = Image.fromarray(rgb).convert('RGB')
        if not look_down:
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
            # image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}.jpg")
        else:
            # image.save(f"{self.save_dir}/debug_raw_{self.episode_idx: 04d}_look_down.jpg")
            pass
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

        text = self.processor.apply_chat_template(self.conversation_history, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[text], images=self.input_images, return_tensors="pt").to(self.device)
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
                past_key_values=self.past_key_values,
                return_dict_in_generate=True,
                raw_input_ids=copy.deepcopy(inputs.input_ids),
            )
        output_ids = outputs.sequences

        t1 = time.time()
        self.llm_output = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        with open(f"{self.save_dir}/llm_output_{self.episode_idx: 04d}.txt", 'w') as f:
            f.write(self.llm_output)
        self.last_output_ids = copy.deepcopy(output_ids[0])
        self.past_key_values = copy.deepcopy(outputs.past_key_values)
        print(f"[S2] output {self.episode_idx}  {self.llm_output} cost: {t1 - t0}s")
        
        if bool(re.search(r'\d', self.llm_output)):
            coord = [int(c) for c in re.findall(r'\d+', self.llm_output)]
            pixel_goal = [int(coord[1]), int(coord[0])]
            image_grid_thw = torch.cat([thw.unsqueeze(0) for thw in inputs.image_grid_thw], dim=0)
            pixel_values = inputs.pixel_values
            t0 = time.time()
            with torch.no_grad():
                traj_latents = self.model.generate_latents(output_ids, pixel_values, image_grid_thw)
                print(f'[S2] generate latents cost: {time.time() - t0}s')
                return None, traj_latents, pixel_goal

        else:
            action_seq = self.parse_actions(self.llm_output)
            return action_seq, None, None

    def step_s1(self, latent, rgb, depth):
        """Stage 1: 轨迹生成"""
        all_trajs = self.model.generate_traj(latent, rgb, depth, use_async=True)
        return all_trajs

    def __del__(self):
        """析构函数：停止异步线程"""
        self.s2_thread_stop_flag = True
        if hasattr(self, 's2_prefetch_thread') and self.s2_prefetch_thread:
            self.s2_prefetch_thread.join(timeout=2.0)
        print("异步S2预执行线程已清理")

