#!/usr/bin/env python3

import gymnasium as gym
import numpy as np
import time
import argparse

def watch_random_policy(env_name="CartPole-v1", num_episodes=5, fps=30, show_info=True):
    """实时观看随机策略的表现"""
    
    # 创建环境（人类可视化模式）
    env = gym.make(env_name, render_mode='human')
    
    # 运行仿真
    all_returns = []
    all_lengths = []
    
    try:
        for episode in range(num_episodes):
            if show_info:
                print(f"=== Episode {episode + 1}/{num_episodes} ===")
            
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            
            while True:
                # 随机选择动作
                action = env.action_space.sample()
                
                # 执行动作
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                
                # 控制帧率
                time.sleep(1.0 / fps)
                
                # 检查是否结束
                if terminated or truncated:
                    break
            
            all_returns.append(total_reward)
            all_lengths.append(steps)
            
            if show_info:
                print(f"Episode 结束:")
                print(f"  总奖励: {total_reward:.2f}")
                print(f"  总步数: {steps}")
                print(f"  最终观察: [{', '.join(f'{x:.3f}' for x in obs)}]")
                
                # 显示当前统计
                current_mean = np.mean(all_returns)
                current_std = np.std(all_returns) if len(all_returns) > 1 else 0
                print(f"  当前平均奖励: {current_mean:.2f} ± {current_std:.2f}")
                print()
            
            # 短暂暂停，让用户看清结果
            if episode < num_episodes - 1:  # 最后一个episode不暂停
                time.sleep(5)
    
    except KeyboardInterrupt:
        print(f"\n用户中断观看，已完成 {len(all_returns)} 个episodes")
    
    # 最终统计
    if all_returns:
        mean_return = np.mean(all_returns)
        std_return = np.std(all_returns)
        mean_length = np.mean(all_lengths)
        max_return = np.max(all_returns)
        min_return = np.min(all_returns)
        
        print(f"\n=== 随机策略观看结果统计 ===")
        print(f"观看的episodes: {len(all_returns)}")
        print(f"平均奖励: {mean_return:.2f} ± {std_return:.2f}")
        print(f"最佳奖励: {max_return:.2f}")
        print(f"最差奖励: {min_return:.2f}")
        print(f"平均步数: {mean_length:.1f}")
        
    env.close()
    return {
        'returns': all_returns,
        'lengths': all_lengths,
        'mean_return': mean_return if all_returns else 0,
        'std_return': std_return if all_returns else 0,
    }

def main():
    parser = argparse.ArgumentParser(description="观看随机策略的实时表现")
    parser.add_argument("--env_name", type=str, default="CartPole-v1",
                        help="环境名称")
    parser.add_argument("--num_episodes", "-n", type=int, default=5,
                        help="观看的episode数量")
    parser.add_argument("--fps", type=int, default=30,
                        help="观看时的帧率（越大越快）")
    parser.add_argument("--show_info", action="store_true", default=True,
                        help="是否显示详细信息")
    
    args = parser.parse_args()
    
    watch_random_policy(
        env_name=args.env_name,
        num_episodes=args.num_episodes,
        fps=args.fps,
        show_info=args.show_info
    )
    
if __name__ == "__main__":
    main()