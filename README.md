# SORT

## 卡尔曼

基于CV模型 (contant velocity model)

量测输入量 $[x,y,s,r]$

状态量 $[x,y,s,r,\dot{x},\dot{y},\dot{s}]$

在雷达中, 由于不依赖于射影空间, 不符合"近大远小", 可以优化为

量测输入量 $[x,y,s,r]$

状态量 $[x,y,s,r,\dot{x},\dot{y}]$

则状态方程
$$
\mathbf{x}_{k+1} = \mathbf{F}\mathbf{x}_{k}+\mathbf{w}_k
$$
状态转移矩阵 $F$
$$
F = 
\begin{bmatrix}
1 & 0 & 0 & 0 & \Delta t & 0 \\
0 & 1 & 0 & 0 & 0 & \Delta t \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$
其中 $\Delta t$ 为时间步长

观测方程中
$$
\mathbf{z} = [x, y, s, r]^T
$$
观测矩阵 $H$
$$
\mathbf{H} = \begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0
\end{bmatrix}
$$


1. 预测 predict
   $$
   \mathbf{x}_{k|k-1} = \mathbf{F} \mathbf{x}_{k-1|k-1}
   $$

   $$
   \mathbf{P}_{k \mid k-1}=\mathbf{F} \mathbf{P}_{k-1 \mid k-1} \mathbf{F}^T+\mathbf{Q}
   $$

   

2. 更新 update

$$
\mathbf{K}_k=\mathbf{P}_{k \mid k-1} \mathbf{H}^T\left(\mathbf{H} \mathbf{P}_{k \mid k-1} \mathbf{H}^T+\mathbf{R}\right)^{-1} 
$$

$$
\mathbf{x}_{k \mid k}=\mathbf{x}_{k \mid k-1}+\mathbf{K}_k\left(\mathbf{z}_k-\mathbf{H} \mathbf{x}_{k \mid k-1}\right) 
$$

$$
\mathbf{P}_{k \mid k}=\left(\mathbf{I}-\mathbf{K}_k \mathbf{H}\right) \mathbf{P}_{k \mid k-1}
$$

(带控制器的卡尔曼滤波还会带上控制矩阵 $\mathbf{B}$ 和控制输入 $\mathbf{u}_k$)



### 参数

#### 观测噪声协方差矩阵 $\mathbf{R}$

观测噪声协方差矩阵 $\mathbf{R}$ 描述了观测数据中的噪声特性。假设观测噪声是白噪声并且各个观测量之间是独立的，$\mathbf{R}$ 可以是一个对角矩阵，其对角线元素表示每个观测量的方差。
$$
\mathbf{R} = \begin{bmatrix} \sigma_x^2 & 0 & 0 & 0 \\ 0 & \sigma_y^2 & 0 & 0 \\ 0 & 0 & \sigma_s^2 & 0 \\ 0 & 0 & 0 & \sigma_r^2 \end{bmatrix}
$$
其中，$\sigma_x^2$, $\sigma_y^2$, $\sigma_s^2$ 和 $\sigma_r^2$ 分别是 $x$、$y$、$s$ 和 $r$ 的观测噪声方差。

#### 状态估计协方差矩阵 $\mathbf{P}$

状态估计协方差矩阵 $\mathbf{P}$ 描述了状态估计的不确定性。它是一个对称正定矩阵，在滤波器初始化时需要设定初值。通常情况下，可以设定较大的初始值表示对初始状态的不确定性较大。
$$
\mathbf{P} = \begin{bmatrix} \sigma_{x_0}^2 & 0 & 0 & 0 & 0 & 0 \\ 0 & \sigma_{y_0}^2 & 0 & 0 & 0 & 0 \\ 0 & 0 & \sigma_{s_0}^2 & 0 & 0 & 0 \\ 0 & 0 & 0 & \sigma_{r_0}^2 & 0 & 0 \\ 0 & 0 & 0 & 0 & \sigma_{u_0}^2 & 0 \\ 0 & 0 & 0 & 0 & 0 & \sigma_{v_0}^2 \end{bmatrix}
$$
其中，$\sigma_{x_0}^2$, $\sigma_{y_0}^2$, $\sigma_{s_0}^2$, $\sigma_{r_0}^2$, $\sigma_{u_0}^2$ 和 $\sigma_{v_0}^2$ 分别是初始状态估计的方差。



#### 过程噪声协方差矩阵 $\mathbf{Q}$

过程噪声协方差矩阵 $\mathbf{Q}$ 描述了系统过程噪声的特性。它反映了模型的不确定性和系统内部的随机扰动。对于一个简单的模型，可以假设过程噪声是白噪声并且各个状态之间是独立的，因此 $\mathbf{Q}$ 也是一个对角矩阵。
$$
\mathbf{Q} = \begin{bmatrix} \sigma_{x_p}^2 & 0 & 0 & 0 & 0 & 0 \\ 0 & \sigma_{y_p}^2 & 0 & 0 & 0 & 0 \\ 0 & 0 & \sigma_{s_p}^2 & 0 & 0 & 0 \\ 0 & 0 & 0 & \sigma_{r_p}^2 & 0 & 0 \\ 0 & 0 & 0 & 0 & \sigma_{u_p}^2 & 0 \\ 0 & 0 & 0 & 0 & 0 & \sigma_{v_p}^2 \end{bmatrix}
$$
其中，$\sigma_{x_p}^2$, $\sigma_{y_p}^2$, $\sigma_{s_p}^2$, $\sigma_{r_p}^2$, $\sigma_{u_p}^2$和 $\sigma_{v_p}^2$ 分别是过程噪声的方差。

调整 $R, P, Q$ 三个矩阵即可, 通常 $P$ 矩阵设大一点, 而 $R$ 和 $Q$ 矩阵中距离和速度维度可设置较大, 而 $s$ 和 $r$ 两个维度可设小一点, 基本不太变化.

## KM

直接使用的 scipy.optimize.linear_sum_assignment 



## 损失

在图片上利用交并比(IoU), 但是在雷达目标跟踪上, 直接定位坐标, 使用欧氏距离即可. 

也许考虑做个归一化.



# 雷达简略版

距离度量直接选择欧式距离

卡尔曼  

* 量测输入量 $[x,y]$
* 状态量 $[x,y,\dot{x},\dot{y}]$

