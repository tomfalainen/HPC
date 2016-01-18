from skimage.io import imread
import matplotlib.pyplot as plt
import libWordDetection as wd
import cv2
import numpy as np
import scipy as sp
import detection_utils as du
import time
raise SystemError
img = imread('001.tif')
img /= 255
t_img = (1 - img).astype(np.ubyte)
#c_range=range(1, 43, 1)
#r_range=range(1, 31, 1)
c_range=range(1, 16, 2)
r_range=range(1, 16, 2)
#c = 2
#r = 3
#k1 = np.ones((r, c), dtype=np.ubyte)

durs = []
for i in range(10):
	start = time.time()
	all_boxes, params = wd.extract_regions(t_img, c_range, r_range)
	durs.append(time.time() - start) 

#s_img2 = cv2.morphologyEx(t_img, cv2.MORPH_CLOSE, k1)
#print np.all(s_img == s_img2)
print np.mean(durs)
durs = []
for i in range(10):
	start = time.time()
	all_boxes2, params2 = du.extract_regions(t_img, c_range, r_range)
	durs.append(time.time() - start) 

print np.mean(durs)

#%%
img = imread('001.tif')
img /= 255
t_img = (1 - img).astype(np.ubyte)
di = sp.io.loadmat('001.tif.mat')
ab = di['all_boxes']
params = di['params']
gtf = "001.tif.dat"
gt_img = np.fromfile(gtf, dtype=np.int32).reshape(img.shape)     
tb = sp.io.loadmat('001gt.tif.mat')['true_boxes']

#ab = ab[:50]
#params = params[:50]

#ab = np.array([ab[1]])
#tb = np.array([tb[6]])

start = time.time()
o1 = wd.calculate_overlap_fg(ab, tb, t_img, gt_img.copy(), params, params)
print time.time() - start
#32.3182749748

start = time.time()
o2 = np.array([[du.calculate_overlap(a, t, t_img, gt_img.copy(), param) for t in tb] for a, param in zip(ab, params)])
print time.time() - start
ys, xs = np.nonzero(np.isclose(o1, o2, 1e-4) == 0)
print o1[1,6], o2[1,6]

#%%
params = np.array([params[1]])
ab = np.array([ab[1]])
tb = np.array([tb[6]])
#%%
print wd.calculate_overlap_fg(ab, tb, t_img, gt_img.copy(), params, params)
print np.array([[du.calculate_overlap(a, t, t_img, gt_img.copy(), param) for t in tb] for a, param in zip(ab, params)])


#%%
tb = np.array([280, 129, 518, 177]).reshape(1, 4)
params = np.array([5, 20]).reshape(1, 2)
ab = tb + 10

b2 = tb[0]
gt = gt_img[b2[1]:b2[3], b2[0]:b2[2]].copy()
gt2 = gt.copy()
gt_val = np.argmax(np.bincount(gt.flatten())[1:]) + 1

gt_valc = wd.get_most_common_value(gt)
print gt_val, gt_valc

gt[gt != gt_val] = 0
gt[gt == gt_val] = 1

gt2 = wd.test_equals_value(gt2, gt_val);
print np.all(gt2 == gt)

start = time.time()
o1 = wd.calculate_overlap_fg(ab, tb, t_img, gt_img.copy(), params, params)
print time.time() - start
print o1


b1, b2 = ab[0], tb[0]
p1 = params[0]
start = time.time()
o2 = np.array([[du.calculate_overlap(a, t, t_img, gt_img.copy(), param) for t in tb] for a, param in zip(ab, params)])
print time.time() - start
print o2
#%%
R, C = p1
tmp_img = img[b1[1]:b1[3], b1[0]:b1[2]].copy()
s_img = cv2.morphologyEx(tmp_img, cv2.MORPH_CLOSE, np.ones((R, C), dtype=np.ubyte))
l_img, n = sp.ndimage.label(s_img)

p_img = l_img.copy()

p2 = p_img.copy()
#proposal box
#zero out excess pixels
mask = img[b1[1]:b1[3], b1[0]:b1[2]]

#p_img[mask == 0] = 0
for r in range(p_img.shape[0]):
    for c in range(p_img.shape[1]):
        if mask[r, c] == 0:
            p_img[r, c] = 0

p_img2 = wd.test_mask_p_img(p2, mask.astype(np.int32))

print np.all(p_img2 == p_img)
#%%

def calculate_overlap(boxes1, boxes2):
    overlaps = np.zeros((len(boxes1), len(boxes2)))
    for i, b1 in enumerate(boxes1):
        for j, b2 in enumerate(boxes2):
                
            #calculate box area
            a1 = (b1[2]- b1[0]) * (b1[3] - b1[1])
            a2 = (b2[2]- b2[0]) * (b2[3] - b2[1])
    
            #find area of intersection
            ai = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0])) * max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
        
            if not ai:
                overlaps[i, j] = 0.0
                continue
            #calculate union    
            au = a1 + a2 - ai
            
            if not au:
                overlaps[i, j] = 0.0
                continue
                
            overlaps[i, j] = float(ai) / float(au)
        
    return overlaps
    
start = time.time()
o1 = wd.calculate_overlap(ab, ab)
print time.time() - start;

start = time.time()
#o2 = np.array([[du.calculate_overlap(a, t) for t in ab] for a in ab])
o2 = calculate_overlap(ab, ab)
print time.time() - start