import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data_rebound(num_frames=200, num_targets=5):
    size = 1
    data = []
    x_lims = 50
    y_lims = 50
    v = 2
    v_scale = 0.5
    np.random.seed(42)  # 固定随机种子
    for target_id in range(num_targets):
        x, y = np.random.randint(0+size, x_lims-size, 2)
        vx, vy = np.random.randint(-v, v, 2)
        for frame in range(num_frames):
            delta_x, delta_y = np.random.normal(0, v_scale, 2)
            x = x + vx + delta_x
            y = y + vy + delta_y
            x1, y1 = x, y
            # x2, y2 = x + size, y + size  # 假设目标的宽高为50
            if x < 0 or x > x_lims:
                vx = -vx
                if x < 0:
                    x = 0
                else:
                    x = x_lims
            if y < 0 or y > y_lims: 
                vy = -vy
                if y < 0:
                    y = 0
                else:
                    y = y_lims
            score = np.random.rand()   # score暂时没用
            data.append([frame + 1, target_id, x1, y1,score])
    return np.array(data)


def generate_synthetic_data_disapper(num_frames=200, num_targets=5):
    size = 1
    data = []
    x_lims = 50
    y_lims = 50
    v = 2
    v_scale = 0.5
    np.random.seed(42)  # 固定随机种子
    for target_id in range(num_targets):
        x, y = np.random.randint(0+size, x_lims-size, 2)
        vx, vy = np.random.randint(-v, v, 2)
        for frame in range(num_frames):
            delta_x, delta_y = np.random.normal(0, v_scale, 2)
            x = x + vx + delta_x
            y = y + vy + delta_y
            x1, y1 = x, y
            if x < 0 or x > x_lims or y < 0 or y > y_lims:
                x, y = np.random.randint(0+size, x_lims-size, 2)
                vx, vy = np.random.randint(-v, v, 2)
            score = np.random.rand()   # score暂时没用
            data.append([frame + 1, target_id, x1, y1,score])
    return np.array(data)
