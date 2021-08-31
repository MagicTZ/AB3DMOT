# Author: Xinshuo Weng
# email: xinshuo.weng@gmail.com

import numpy as np
# from sklearn.utils.linear_assignment_ import linear_assignment    # deprecated
from scipy.optimize import linear_sum_assignment
from AB3DMOT_libs.bbox_utils import convert_3dbox_to_8corner, iou3d
from AB3DMOT_libs.kalman_filter import KalmanBoxTracker

def associate_detections_to_trackers(detections, trackers, iou_threshold=0.01):   
	"""
	Assigns detections to tracked object (both represented as bounding boxes)

	detections:  N x 8 x 3
	trackers:    M x 8 x 3


	Returns 3 lists of matches, unmatched_detections and unmatched_trackers
	"""
	# 如果轨迹没有找到，则所有detection都是unmatched_detections
	if (len(trackers)==0): 
		return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 8, 3), dtype=int)
	iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

	# 计算IoU
	for d, det in enumerate(detections):
		for t, trk in enumerate(trackers):
			# 1. 计算2D IoU: 在鸟瞰图下(x-z轴)，通过多边形裁剪(Sutherland-Hodgman算法)的方法，找到交点坐标，然后使用convex-hull计算2d Iou
			# 2. 根据y轴方向的坐标差计算高度，从而计算出3D IoU
			iou_matrix[d, t] = iou3d(det, trk)[0]             # det: 8 x 3, trk: 8 x 3
	# matched_indices = linear_assignment(-iou_matrix)      # hougarian algorithm, compatible to linear_assignment in sklearn.utils

	# 匈牙利算法匹配，得到返回结果中的matched，利用scipy.optimize中的linear_sum_assignment函数，该函数求得是最小代价
	# 求最大IoU，先将IoU取负，相应的索引位置是(row_ind, col_ind)
	row_ind, col_ind = linear_sum_assignment(-iou_matrix)      # hougarian algorithm
	matched_indices = np.stack((row_ind, col_ind), axis=1)     # (n,2)，第1列：detection，第2列：tracking

	unmatched_detections = []
	for d, det in enumerate(detections):
		if (d not in matched_indices[:, 0]): unmatched_detections.append(d)
	unmatched_trackers = []
	for t, trk in enumerate(trackers):
		if (t not in matched_indices[:, 1]): unmatched_trackers.append(t)

	# filter out matched with low IOU
	# 过滤掉IOU过低的匹配结果，并将其加入unmatched_detections和unmatched_tracker中
	matches = []  # 最终匹配成功的结果
	for m in matched_indices:
		if (iou_matrix[m[0], m[1]] < iou_threshold):
			unmatched_detections.append(m[0])
			unmatched_trackers.append(m[1])
		else: matches.append(m.reshape(1, 2))
	if (len(matches) == 0): 
		matches = np.empty((0, 2), dtype=int)
	else: matches = np.concatenate(matches, axis=0)

	return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class AB3DMOT(object):			  # A baseline of 3D multi-object tracking
	def __init__(self, max_age=2, min_hits=3):      # max age will preserve the bbox does not appear no more than 2 frames, interpolate the detection
		"""
		Sets key parameters for SORT                
		"""
		self.max_age = max_age  # death age
		self.min_hits = min_hits  # birth age
		self.trackers = []
		self.frame_count = 0
		self.reorder = [3, 4, 5, 6, 2, 1, 0]
		self.reorder_back = [6, 5, 4, 0, 1, 2, 3]

	def update(self, dets_all):
		"""
		Params:
		  dets_all: dict
			dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]
			info: a array of other info for each det
		Requires: this method must be called once for each frame even with empty detections.
		Returns the a similar array, where the last column is the object ID.

		NOTE: The number of objects returned may differ from the number of detections provided.
		"""
		dets, info = dets_all['dets'], dets_all['info']         # dets: N x 7, float numpy array

		# reorder the data to put x,y,z in front to be compatible with the state transition matrix
		# where the constant velocity model is defined in the first three rows of the matrix
		dets = dets[:, self.reorder]					# reorder the data to [[x,y,z,theta,l,w,h], ...]
		self.frame_count += 1

		trks = np.zeros((len(self.trackers), 7))         # N x 7 , # get predicted locations from existing trackers.
		to_del = []		# 要被删除的轨迹
		ret = []		# 追踪结果，返回值，用于最后的追踪结果保存

		# 得到每一条轨迹的预测结果
		for t, trk in enumerate(trks):
			pos = self.trackers[t].predict().reshape((-1, 1))
			trk[:] = [pos[0], pos[1], pos[2], pos[3], pos[4], pos[5], pos[6]]       
			if np.any(np.isnan(pos)):		# 如果该时刻轨迹属性不存在，就它的时刻其存入待删除的轨迹
				to_del.append(t)
		trks = np.ma.compress_rows(np.ma.masked_invalid(trks))		# 使用numpy.ma来支持掩码数组, 仅保存完整的预测结果（将无效轨迹从trks中去除）
																	# 如 [[nan, 1, 2], [3, nan, 5], [6, 7, 8]] 结果就变成了 [[6, 7, 8]]
		# 根据to_del list中的索引删除轨迹
		for t in reversed(to_del):
			self.trackers.pop(t)

		# 匈牙利匹配： detection vs. tracking (8 corners) 维度：8*3
		dets_8corner = [convert_3dbox_to_8corner(det_tmp) for det_tmp in dets]
		if len(dets_8corner) > 0: dets_8corner = np.stack(dets_8corner, axis=0)
		else: dets_8corner = []
		trks_8corner = [convert_3dbox_to_8corner(trk_tmp) for trk_tmp in trks]
		if len(trks_8corner) > 0: trks_8corner = np.stack(trks_8corner, axis=0)
		# 使用匈牙利算法对tracking和detection的结果进行匹配，返回：成功匹配，未成功匹配的检测，未成功匹配的轨迹结果
		matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets_8corner, trks_8corner)

		# update matched trackers with assigned detections
		for t, trk in enumerate(self.trackers):
			if t not in unmatched_trks:
				d = matched[np.where(matched[:, 1] == t)[0], 0]     # a list of index
				trk.update(dets[d, :][0], info[d, :][0])		# 更新匹配上的轨迹（观测+预测）

		# create and initialise new trackers for unmatched detections
		for i in unmatched_dets:        # a scalar of index
			trk = KalmanBoxTracker(dets[i, :], info[i, :])   # 创建新的轨迹
			self.trackers.append(trk)		# 加入到轨迹的集合中
		i = len(self.trackers)
		for trk in reversed(self.trackers):
			d = trk.get_state()      # bbox location
			d = d[self.reorder_back]			# change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]

			# 确定最终要返回的tracking，处理轨迹的birth和death
			# death：已经超过max_age帧没有更新的轨迹，每个tracking都使用time_since_update进行记录
			# birth：已经连续更新了min_hits帧，每个tracking都使用hits参数记录或目前已经匹配的帧的数目小于hits，比如第1帧
			if ((trk.time_since_update < self.max_age) and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits)):      
				ret.append(np.concatenate((d, [trk.id + 1], trk.info)).reshape(1, -1)) # +1 as MOT benchmark requires positive
			i -= 1

			# remove dead tracklet
			if (trk.time_since_update >= self.max_age): 
				self.trackers.pop(i)
		if (len(ret) > 0): return np.concatenate(ret)			# h,w,l,x,y,z,theta, ID, other info, confidence
		return np.empty((0, 15))    