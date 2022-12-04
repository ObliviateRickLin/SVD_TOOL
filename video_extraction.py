import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
# import scipy
H, W = None, None
path = '/Users/zhaosonglin/Desktop/test_videos/640_360/walking.mp4'
interval = 20

def read_video(path, interval):
    global H
    global W
    capture = cv.VideoCapture(path)
    print('file opened:', capture.isOpened())  # return True if file can be opened
    count = 0
    l = [0,0,0]
    if capture.isOpened():
        open, frame = capture.read()
        H, W = frame.shape[:-1]
    else:
        open = False

    while open:
        ret, frame = capture.read() # return two values: (bool, numpy.ndarray) 
        if frame is None:
            break
        if ret == True:
            if count % interval == 0:
                for i in range(3):
                    m = frame[:,:,i].flatten('F')
                    if count == 0:
                        l[i] = m
                    else:
                        l[i] = np.vstack((l[i], m))  
            # press q to exit
            count += 1
            # if cv.waitKey(1) & 0xFF == ord('q'):
            #     break
    # capture.release()
    return l

def extract_scipy(a):
    u, s, v = svds(a)
    u1 = u[:,-1]
    s1 = s[-1]
    v1 = v[-1]
    m = s1 * np.outer(u1, v1)
    eigen_bg = m[:,0]
    return eigen_bg

def extract(a):
    u, s, v = svd(a)
    m = s * np.outer(u, v)
    eigen_bg = m[:,0]
    return eigen_bg

def svd(A):
    v, eigv = power_method(A.T@A)
    sigma = eigv ** (1/2)
    u = A@v / sigma
    return u, sigma, v

def power_method(A, threshold = 10**(-15)):
    row, col = A.shape
    start = [1,]
    for _ in range(row-1):
        start.append(0)
    x = np.array(start).T
    sigma, sigma_ = 1, 10
    while abs(sigma - sigma_) >= threshold:
        x_= A@x
        x = x_ / np.linalg.norm(x_)
        sigma_, sigma = sigma, x.T @ A @ x
    return x, sigma
    
def extract_bg(path, interval = 5):
    #print('---creating matrix A---')
    L = read_video(path, interval)
    A1 = L[0].T.astype(float)
    A2 = L[1].T.astype(float)
    A3 = L[2].T.astype(float)
    #print('---extracting background---')
    img1 = extract(A1)
    img2 = extract(A2)
    img3 = extract(A3)
    print('---creating background image---')
    img1 = np.reshape(img1, (H, W), 'F')
    img2 = np.reshape(img2, (H, W), 'F')
    img3 = np.reshape(img3, (H, W), 'F')
    img1[img1>255] = 255
    img1[img1<0] = 0
    img2[img2>255] = 255
    img2[img2<0] = 0
    img3[img3>255] = 255
    img3[img3<0] = 0
    res = np.zeros((H,W,3))
    res[:,:,0] = img3 
    res[:,:,1] = img2 
    res[:,:,2] = img1 
    res = np.uint8(res)
    return res
    #plt.imshow(res)
    #plt.show()
