import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
import time
from Sort import Sort, KalmanBoxTracker, associate_detections_to_trackers
import generate 

# 主测试函数
def main_test():
    np.random.seed(42)
    num_frames = 200
    num_targets = 6
    num_size = 1
    # 生成仿真数据
    dets = generate.generate_synthetic_data_rebound(num_frames, num_targets)
    #dets = generate.generate_synthetic_data_disapper(num_frames, num_targets)

    # 设置显示参数
    display = True
    history_display = True
    colours = np.random.rand(32, 3)  # 用于显示

    if display:
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    total_time = 0.0
    total_frames = 0
    history = {}

    mot_tracker = Sort(max_age=1, min_hits=3, distance_threshold=20)  # 创建 SORT 跟踪器实例

    n_frame = int(dets[:, 0].max())

    for frame in range(1, int(dets[:, 0].max()) + 1):
        # 提取当前帧的检测
        frame_dets = dets[dets[:, 0] == frame, 2:5]
        ground_truth = dets[dets[:, 0] == frame, 1:5]
        
        total_frames += 1

        if display:
            ax1.clear()
            plt.title('Multi-Object Tracking '+'frame:'+ str(frame) + " / " + str(n_frame) + '  number of object :' + str(mot_tracker.num_objext))
            ax1.set_xlim(-5, 55)
            ax1.set_ylim(-5, 55)

        mot_tracker.num_objext = 0
        start_time = time.time()
        trackers = mot_tracker.update(frame_dets)       
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
            # print('%d,%d,%.2f,%.2f,1,-1,-1,-1' % (frame, d[2], d[0], d[1]))
            if display:
                d = d.astype(np.int32)
                if history_display:
                    track_id = d[2]
                    if track_id not in history:
                        history[track_id] = []
                    history[track_id].append((d[0], d[1]))

                # ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3, ec=colours[d[4]%32, :]))
                ax1.add_patch(patches.Rectangle((d[0], d[1]), num_size ,num_size, fill=False, lw=3, ec=colours[d[2]%32,:]))
                
                if history_display:
                    track_points = np.array(history[track_id])
                    ax1.plot(track_points[:, 0], track_points[:, 1], '-', c=colours[track_id % 32, :])
                # ax1.plot(history[d[2]][0],history[d[2]][1],'-',c=colours[d[2]%32,:])

        for e in ground_truth:
            if display:
                ax1.text(e[1], e[2], str(int(e[0])), color='red')

        if display:
            fig.canvas.flush_events()
            plt.draw()
            plt.savefig('./result3/'+str(frame)+'.png')
            # time.sleep(0.02)  
            # 增加一点延迟以便观察

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

    if display:
        print("Note: to get real runtime results run without the option: --display")

main_test()
