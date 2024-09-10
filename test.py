import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import time
from Sort import Sort, KalmanBoxTracker, associate_detections_to_trackers

# 生成仿真轨迹数据
def generate_synthetic_data(num_frames=200, num_targets=5):
    data = []
    np.random.seed(42)  # 固定随机种子
    for target_id in range(num_targets):
        size = 20
        x, y = np.random.randint(0+size, 500-size, 2)
        vx, vy = np.random.randint(-5, 5, 2)
        for frame in range(num_frames):
            x += vx
            y += vy
            x1, y1 = x, y
            # x2, y2 = x + size, y + size  # 假设目标的宽高为50
            if x < 0 or x > 500: vx = -vx
            if y < 0 or y > 500: vy = -vy
            score = np.random.rand()   # score暂时没用
            data.append([frame + 1, target_id + 1, x1, y1,score])
    return np.array(data)

# 原始轨迹可视化
def visualize_trajectories(data, num_frames=200, num_targets=5,num_size = 20):
    colours = np.random.rand(num_targets, 3)  # 生成随机颜色
    plt.ion()
    fig, ax = plt.subplots()

    for frame in range(1, num_frames + 1):
        ax.clear()
        plt.title('Synthetic Data Trajectories')
        ax.set_xlim(-50, 550)
        ax.set_ylim(-50, 550)

        frame_data = data[data[:, 0] == frame]
        for target_id in range(1, num_targets + 1):
            target_data = frame_data[frame_data[:, 1] == target_id]
            if len(target_data) > 0:
                x1, y1, = target_data[0, 2:4]
                ax.add_patch(plt.Rectangle((x1, y1), num_size ,num_size , fill=True, color=colours[target_id - 1], alpha=0.5))
                ax.text(x1, y1, str(target_id), color=colours[target_id - 1])

        plt.draw()
        plt.pause(0.1)

    plt.ioff()
    plt.show()

# 主测试函数
def main_test():
    np.random.seed(42)
    num_frames = 200
    num_targets = 10
    num_size = 20
    # 生成仿真数据
    dets = generate_synthetic_data(num_frames, num_targets)
    
