import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#from scipy.sparse.linalg import gemm
import scipy.linalg.blas as FB
from scipy.sparse.linalg import svds
import time
import math

interval = 20



def read_video(path, interval=30):
    '''
    return: list l with ith element being the A.T, (frame x H*W)
    '''
    global H
    global W
    capture = cv.VideoCapture(path)
    frame_count = capture.get(cv.CAP_PROP_FRAME_COUNT)
    # print(frame_count)
    if interval == 1:
        size = int(frame_count - 1)
    else:
        size = math.ceil(frame_count/interval) 
    count = 0
    num = 0

    if capture.isOpened():
        open, frame = capture.read()
        H, W = frame.shape[:-1]
    else:
        open = False
    l = [np.zeros((size,H*W)),np.zeros((size,H*W)),np.zeros((size,H*W))]
    while open:
        ret, frame = capture.read() # return two values: (bool, numpy.ndarray) 
        if frame is None:
            break
        if ret == True:
            if count % interval == 0:
                for i in range(3):
                    m = frame[:,:,i].flatten('F')
                    l[i][num] = m
                num += 1
            count += 1 
    return l


def extract_background_scipy(A):
    '''
    return: flatten matrix of the background (H*W x 1)
    '''
    u, s, v = svds(A, k=1)
    # print(s)
    u1 = u[:,-1]
    s1 = s[-1]
    v1 = v[-1]
    # print(s[-2:])
    B = s1 * v1[0] * u1
    # B = s * v[0] * u
    return B


def extract_background(A, threshold = 10**(-14)):
    # print(A.shape)
    a = time.time()
    M = A.T @ A
    b = time.time()
    # print('xxx',b-a)
    w = A.shape[1] # width of A i,e, # of frames selected
    x = [0] * w
    v = np.full_like(x, (1/w)**0.5, dtype=np.double)
    sigma_sq = v.T @ M @ v
    sigma_sq_ = 0
    i = 0
    print(sigma_sq - sigma_sq_)
    while abs(sigma_sq - sigma_sq_) >= threshold:
        print(sigma_sq - sigma_sq_)
        i += 1
        # print(i)
        v_= M@v
        v = v_ / np.linalg.norm(v_)
        sigma_sq_, sigma_sq = sigma_sq, v.T @ M @ v
    B = v[0] * (A @ v)
    print('------')
    return B
    
def extract_bg(path, interval=20):
    L = read_video(path, interval)
    A1 = L[0].T.astype(float)
    A2 = L[1].T.astype(float)
    A3 = L[2].T.astype(float)
    num = A1.shape[1]
    a = time.time()
    img1 = extract_background_scipy(A1)
    img2 = extract_background_scipy(A2)
    img3 = extract_background_scipy(A3)
    b = time.time()
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
    
