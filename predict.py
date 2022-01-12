
from __future__ import division
import sys
import os


import numpy as np
import timeit
import pickle
from numpy.ma import asarray
from scipy.signal import convolve2d
from sklearn import svm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


test_dir=str(sys.argv[1])
out_dir=str(sys.argv[2])


preprocessed_images = []


def prePro(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

def lpq(img,winSize=3,freqestim=1,mode='nh'):
    rho=0.90

    STFTalpha=1/winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    sigmaS=(winSize-1)/4 # Sigma for STFT Gaussian window (applied if freqestim==2)
    sigmaA=8/(winSize-1) # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)

    convmode='valid' # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img=np.float64(img) # Convert np.image to double
    r=(winSize-1)/2 # Get radius from window size
    x=np.arange(-r,r+1)[np.newaxis] # Form spatial coordinates in window

    if freqestim==1:  #  STFT uniform window
        #  Basic STFT filters
        w0=np.ones_like(x)
        w1=np.exp(-2*np.pi*x*STFTalpha*1j)
        w2=np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1=convolve2d(convolve2d(img,w0.T,convmode),w1,convmode)
    filterResp2=convolve2d(convolve2d(img,w1.T,convmode),w0,convmode)
    filterResp3=convolve2d(convolve2d(img,w1.T,convmode),w1,convmode)
    filterResp4=convolve2d(convolve2d(img,w1.T,convmode),w2,convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)

    ## Switch format to uint8 if LPQ code np.image is required as output
    if mode=='im':
        LPQdesc=np.uint8(LPQdesc)

    ## Histogram if needed
    if mode=='nh' or mode=='h':
        LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]

    ## Normalize histogram if needed
    if mode=='nh':
        LPQdesc=LPQdesc/LPQdesc.sum()

    return LPQdesc


prepro_recorded_times=[]

for filename in os.listdir(test_dir):
        start_time=timeit.default_timer()
        image = prePro(test_dir+filename)
        ##print(test_dir+filename)
        prepro_recorded_times.append(timeit.default_timer()-start_time)
        preprocessed_images.append(image)
        
lpq_recorded_times=[]
lpq_images=[]

for i in range(len(preprocessed_images)):
    start_time=timeit.default_timer()
    image_FE = lpq(preprocessed_images[i])
    lpq_recorded_times.append(timeit.default_timer()-start_time)
    lpq_images.append(image_FE)



loaded_model = pickle.load(open('finalized_model.sav', 'rb'))


predicted_class=[]
prediction_recorded_times=[]

file2=open(out_dir+'results.txt','a')


for i in range(len(lpq_images)):
    
    single_lpq_image=[]
    single_lpq_image.append(lpq_images[i])
    start_time=timeit.default_timer()
    predicted_output=loaded_model.predict(single_lpq_image)
    prediction_recorded_times.append(timeit.default_timer()-start_time)
    predicted_class.append(predicted_output)
    if(i!=len(lpq_images)-1):
        file2.write(str(predicted_output[0])+'\n')
    else:
        file2.write(str(predicted_output[0]))
    
file2.close()  


total_time=np.asarray(prediction_recorded_times)+np.asarray(lpq_recorded_times)+np.asarray(prepro_recorded_times)

file1 = open(out_dir+'times.txt', 'a')
for i in range(len(total_time)):
    output_time=round(total_time[i], 2)
    if(output_time==0):
        output_time=0.001
    if(i!=len(total_time)-1):
        file1.write(str(output_time)+'\n')
    else:
        file1.write(str(output_time))
file1.close()






