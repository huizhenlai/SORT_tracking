"""
		SORT: A Simple, Online and Realtime Tracker
		Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

		This program is free software: you can redistribute it and/or modify
		it under the terms of the GNU General Public License as published by
		the Free Software Foundation, either version 3 of the License, or
		(at your option) any later version.

		This program is distributed in the hope that it will be useful,
		but WITHOUT ANY WARRANTY; without even the implied warranty of
		MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
		GNU General Public License for more details.

		You should have received a copy of the GNU General Public License
		along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from __future__ import print_function

import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io

import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)

def linear_assignment(cost_matrix): # 实现匈牙利匹配
	try:
		import lap
		_, x, y = lap.lapjv(cost_matrix, extend_cost=True)
		return np.array([[y[i],i] for i in x if i >= 0]) #
	except ImportError:
		from scipy.optimize import linear_sum_assignment
		x, y = linear_sum_assignment(cost_matrix)
		return np.array(list(zip(x, y)))

def distance_batch(test, gt):
	gt = np.expand_dims(gt, 0)
	test = np.expand_dims(test, 1)

	o = np.sqrt((gt[...,0] - test[...,0])**2 + (gt[...,1] - test[...,1])**2)
	return (o)

def convert_to_1(radar_position):
	x = radar_position[0]
	y = radar_position[1]
	return np.array([x, y]).reshape((2, 1))

def convert_to_2(radar_position):
	x = radar_position[0]
	y = radar_position[1]
	return np.array([x, y]).reshape((1, 2))


class KalmanBoxTracker(object):
	"""
	This class represents the internal state of individual tracked objects observed as bbox.
	"""
	count = 0
	def __init__(self,radar_position):
		"""
		Initialises a tracker using initial bounding box.
		"""
		#define constant velocity model
		self.kf = KalmanFilter(dim_x=4, dim_z=2)  
		# 7个状态变量，4个观测输入, 这里的观测输入为中心表示法的[x,y,s,r] 
		# 状态量为 [x,y,s,r,x',y',s'] 这里的问题在于 为什么会认为面积量s是一个变化量
		# !因为原论文用来表示的是图片的信息, 如果目标靠近摄像机, 目标就会变大, 而横纵比大概率是恒定的
		# https://github.com/abewley/sort/issues/91 提到了这个问题
		# 对于雷达的话, 还是观测包含速度 但是先不管速度吧

		self.kf.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
		self.kf.H = np.array([[1,0,0,0],[0,1,0,0]])

		self.kf.P[2:,2:] *= 100. # give high uncertainty to the unobservable initial velocities
		self.kf.P *= 10.

		self.kf.Q[2,2] *= 0.1
		self.kf.Q[3,3] *= 0.1
		self.kf.x[:2] = convert_to_1(radar_position)  # ! 这里是重点 x的维度是 [4,1] 这个地方实际上只是给一个初始值

		# F是状态变换模型，H是观测函数，R为测量噪声矩阵，P为协方差矩阵，Q为过程噪声矩阵。 R : (2,2) F (4,4) H (2,4)

		self.time_since_update = 0
		self.id = KalmanBoxTracker.count
		KalmanBoxTracker.count += 1
		self.history = []
		self.hits = 0
		self.hit_streak = 0
		self.age = 0

	def update(self,radar_position):
		"""
		Updates the state vector with observed bbox.
		"""
		self.time_since_update = 0
		self.history = []
		self.hits += 1
		self.hit_streak += 1
		self.kf.update(convert_to_1(radar_position))
		#中心表示法

	def predict(self):
		"""
		Advances the state vector and returns the predicted bounding box estimate.
		"""
		
		self.kf.predict()
		self.age += 1
		if(self.time_since_update>0):
			self.hit_streak = 0
		self.time_since_update += 1
		self.history.append(convert_to_2(self.kf.x))
		return self.history[-1]

	def get_state(self):
		"""
		Returns the current bounding box estimate.
		"""
		return convert_to_2(self.kf.x)

def associate_detections_to_trackers(detections,trackers, distance_threshold):

	"""
	将检测框关联到跟踪目标（objects）或者轨迹（tracks），而不是跟踪器（trackers）
	Assigns detections to tracked object (both represented as bounding boxes)

	Returns 3 lists of matches, unmatched_detections and unmatched_trackers
	"""
	if(len(trackers)==0):
		return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,3),dtype=int)

	distance_matrix = distance_batch(detections, trackers)
	if min(distance_matrix.shape) > 0:
		a = (distance_matrix < distance_threshold).astype(np.int32)
		if a.sum(1).max() == 1 and a.sum(0).max() == 1:
			matched_indices = np.stack(np.where(a), axis=1)
		else:
			matched_indices = linear_assignment(distance_matrix)
	else:
		matched_indices = np.empty(shape=(0,2))

	unmatched_detections = []
	for d, det in enumerate(detections):
		if(d not in matched_indices[:,0]):
			unmatched_detections.append(d)
	
	unmatched_trackers = []
	for t, trk in enumerate(trackers):
		if(t not in matched_indices[:,1]):
			unmatched_trackers.append(t)
	# 找到没有匹配的检测框和跟踪器

	#filter out matched with low IOU
	matches = []
	for m in matched_indices:
		if(distance_matrix[m[0], m[1]]>distance_threshold):
			unmatched_detections.append(m[0])
			unmatched_trackers.append(m[1])
		else:
			matches.append(m.reshape(1,2))
	if(len(matches)==0):
		matches = np.empty((0,2),dtype=int)
	else:
		matches = np.concatenate(matches,axis=0)

	return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
	# KM匹配之后，再删掉阈值小于0.3的匹配。

class Sort(object):
	def __init__(self, max_age=1, min_hits=3, distance_threshold=10):
		# Sort 类主要负责管理多个 KalmanBoxTracker 实例，通过关联检测框和跟踪器来跟踪物体。
		# 它使用卡尔曼滤波器进行状态预测和更新，通过 IoU 进行检测和跟踪器的关联，
		# 并在一定条件下初始化新的跟踪器或移除失效的跟踪器

		"""
		Sets key parameters for SORT
		"""
		self.num_objext = 0
		self.max_age = max_age
		# 允许一个对象未被检测到的最大帧数 有一帧没检测到就终止跟踪
		self.min_hits = min_hits
		# 最小命中次数，只有达到这个数值的跟踪器才会被视为一个有效的对象。 所谓“试用期”，防止跟踪误报
		self.distance_threshold = distance_threshold
		self.trackers = []
		# 所有跟踪器列表
		self.frame_count = 0
		self.history = []

	def update(self, dets=np.empty((0, 3))):
		"""
		Params:
			dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
		Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
		Returns the a similar array, where the last column is the object ID.

		NOTE: The number of objects returned may differ from t he number of detections provided.
		"""
		self.frame_count += 1
		# get predicted locations from existing trackers.
		trks = np.zeros((len(self.trackers), 3))
		to_del = []
		ret = []
		for t, trk in enumerate(trks):
			pos = self.trackers[t].predict()[0]
			trk[:] = [pos[0], pos[1], 0]
			# score 初始化为0
			if np.any(np.isnan(pos)):
				to_del.append(t)
		trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
		for t in reversed(to_del):
			self.trackers.pop(t)
		# 如果有NaN等无效值, 删除

		matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks, self.distance_threshold)

		# update matched trackers with assigned detections
		for m in matched:
			self.trackers[m[1]].update(dets[m[0], :])
		# 有关联的每一个跟踪器都update

		# create and initialise new trackers for unmatched detections
		for i in unmatched_dets:
			trk = KalmanBoxTracker(dets[i,:])
			self.trackers.append(trk)
			# self.history.append([])
			# 为每一个失配的检测目标初始化一个跟踪器, 添加进trackers中

		i = len(self.trackers)
		for trk in reversed(self.trackers):
			d = trk.get_state()[0]
			if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
				ret.append(np.concatenate((d,[trk.id])).reshape(1,-1)) # +1 as MOT benchmark requires positive
				self.num_objext += 1
			i -= 1
			# 如果time_since_update最近有更新(=0), 且 (命中次数 <= 最小命中次数 or 帧数<= 最小命中次数)
			# 后者条件的意思是, 在前几帧的情况下, 直接进行匹配?
			# trk.id 应该是个数?

			# remove dead tracklet
			if(trk.time_since_update > self.max_age):
				self.trackers.pop(i)
				self.num_objext -= 1
			# 如果跟踪器的未更新时间超过最大允许时间，则将其移除。

		if(len(ret)>0):
			return np.concatenate(ret)
		return np.empty((0,3))