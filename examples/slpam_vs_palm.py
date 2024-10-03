import sys
sys.path.insert(0, '../python_dms/lib/')
# from dms_new import *
from tools_dms import *
# from tools_dms_cuda import *

from PIL import Image
import cv2
import scipy.io
import math

# data = scipy.io.loadmat('../degraded_images/dots_52_v3_noise_0.05_blur_1_1_3.mat')
# image_np    = data['f']
# image_np_noisy 	 = data['fNoisy']
# exact_contour = data['e_exacte']


image = cv2.imread('10081.jpg', cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
# # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_np = np.asarray(image, dtype="int32" )
size= 300
image_np = np.asarray(image, dtype="int32" )[0:size,0:size]/255.
np.random.seed(1)
image_np_noisy = (image_np + 0.1*np.random.normal(0,1,size=image_np.shape))
# plt.figure()
# plt.imshow(image_np_noisy)
# plt.show()
A = np.ones((image_np_noisy.shape[0],image_np_noisy.shape[1]))
# 
# method='SLPAM'
method='SLPAM-eps-descent'
# method='PALM-eps-descent'
# norm_type='AT'
# method='PALM'
norm_type='AT'
# norm_type='l1'
eps=2
eps_AT_min=0.002

time_start= time.time()
model = DMS(beta=3.485,lamb=0.00205,
            method=method,MaximumIteration=1,
            noised_image_input=image_np_noisy, norm_type=norm_type,stop_criterion=1e-4,
            dk_SLPAM_factor=1e-4,optD='OptD',eps=eps,eps_AT_min=eps_AT_min,A=A)

out = model.process()
print('time:  ',time.time()-time_start)
cont =out[0] 

def edges_to_pix_contour(cont_edges):
    pix = cont_edges[:,:,0]+cont_edges[:,:,1]
    return np.clip(pix,0,1)

cont_thres = np.ones_like(cont)*(cont>0.1)

print(PSNR(out[1],image_np))
# print(jaccard(cont_thres,exact_contour))
plt.figure()
plt.subplot(321)
plt.imshow(out[1],'gray')
plt.subplot(322)
plt.imshow(image_np_noisy,'gray')
plt.subplot(323)
plt.imshow(edges_to_pix_contour(cont),'gray')
plt.subplot(324)
plt.imshow(edges_to_pix_contour(cont)>eps_AT_min,'gray')
# plt.hist(cont[:,:,0].flatten())
plt.subplot(325)
plt.plot(out[2],label=method)
plt.legend()
plt.show()

descent_factor= 1.5
sizeplot = int(np.ceil(math.log(eps/eps_AT_min,descent_factor)))
plt.figure()
for i in range(sizeplot):
    plt.subplot(3,sizeplot,i+1)
    plt.imshow(out[3][i],'gray')
    plt.subplot(3,sizeplot,i+sizeplot+1)
    plt.imshow(edges_to_pix_contour(out[4][i]),'gray')
    # plt.subplot(3,sizeplot,i+2*sizeplot+1)
plt.show()

