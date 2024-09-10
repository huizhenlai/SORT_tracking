import cv2
import os

# 设置输入图片的路径
input_path = './result2'

# 设置输出视频的路径和名称
output_path = 'result2_video.mp4'

# 设置视频的参数
fps = 15  # 帧率
size = (640, 480)  # 视频分辨率

# 创建视频写入对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, size)

# 遍历输入图片,并将其添加到视频中
for i in range(200):
    img_path = os.path.join(input_path, f'{i+1}.png')
    img = cv2.imread(img_path)
    out.write(img)

# 释放视频写入对象
out.release()

print('Video creation complete!')