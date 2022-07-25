import cv2
from cv2 import reduce
import numpy as np
#import os # remove before submit
#import time #remove before submit

def task1_2(src_path, clean_path, dst_path):
    """
    This is main function for task 1.
    It takes 3 arguments,
    'src_path' is path for source image.
    'clean_path' is path for clean image.
    'dst_path' is path for output image, where your result image should be saved.

    You should load image in 'src_path', and then perform task 1-2,
    and then save your result image to 'dst_path'.
    """
    #print("task1_2")
    noisy_img = cv2.imread(src_path)
    clean_img = cv2.imread(clean_path)
    result_img = None
    # # # do noise removal
    sigma_s = 20 #need to change
    sigma_r = 65 #need to change (65 for snow) (40 for fox)
    
    window_size = 100 
    b_rms = 2000 #best rms
    tmp_filter = 0
    filter = {0:'median', 1:'bilateral', 2: 'my(average)'}

    for i in range(0, 9):
        j = i%3
        kernel_size = 2*j+3 #3,5,7
        if(0<=i<=2): #MEDIAN FILTER
            tmp = apply_median_filter(noisy_img, kernel_size)
            rms = calculate_rms(clean_img, tmp)
            k = 0
            #print("rms now is {}".format(rms))
        elif(3<=i<=5): #BILATERAL FILTER
            tmp = apply_bilateral_filter(noisy_img, kernel_size, sigma_s, sigma_r)
            rms = calculate_rms(clean_img, tmp)
            k = 1
            #print("rms now is {}".format(rms))
        else: #MY(average) FILTER
            tmp = apply_my_filter(noisy_img, kernel_size)
            rms = calculate_rms(clean_img, tmp)
            k = 2
            #print("rms now is {}".format(rms))
        if (rms<b_rms):
            result_img = tmp.copy()
            b_rms = rms
            window_size = kernel_size
            tmp_filter = k
    print("optimal filter is {}".format(filter[tmp_filter]))
    print("window_size is {}".format(window_size))
    print("best rms is {}".format(b_rms))
    #print("b_rms is {}".format(b_rms))
    #print("Best RMS:", calculate_rms(clean_img, result_img)) 
    cv2.imwrite(dst_path, result_img)
    pass



def apply_median_filter(img, kernel_size): #useful for salt and pepper noise
    """
    You should implement median filter using convolution in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is an int value, which determines kernel size of median filter.

    You should return result image.
    """
    #print(img.shape)
    p = int((kernel_size-1)/2) #number of padding
    w = int((kernel_size + 1) / 2 )#use to move index in output image
    k = int(kernel_size*kernel_size / 2) #index for median

    #get median 
    new_shape0 = img.shape[0]+2*p
    new_shape1 = img.shape[1]+2*p
    #image_output = np.zeros((new_shape0,new_shape1,img.shape[2])) #padding #img.copy() #np.zeros(img.shape)
    #image_output = np.full((new_shape0,new_shape1,img.shape[2]), 0.5)
    image_output = np.ones((new_shape0,new_shape1,img.shape[2]))
    image_output[p:new_shape0-p, p:new_shape1-p] = img.copy()
    img = image_output.copy()
    for x in range(0, new_shape0-kernel_size+1):
        for y in range(0, new_shape1-kernel_size+1):
            for c in range(0, img.shape[2]):
                #print("x, y is {} {}".format(x, y))
                tmp = (img[x:x+kernel_size, y:y+kernel_size,c]).copy()
                #print(tmp)
                t = tmp.flatten() #flatten -> array
                # print(t)
                t.sort()
                median = t[k]
                #print("median is {}".format(median))
                image_output[int(x+w-1),int(y+w-1),c] = median 
                #print(image_output)
    #print("happy")
    img = image_output[p:new_shape0-p, p:new_shape1-p]
    #print(img.shape)
    return img
    
    
    


def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    """
    You should implement bilateral filter using convolution in this function.
    It takes at least 4 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is a int value, which determines kernel size of average filter.
    'sigma_s' is a int value, which is a sigma value for G_s(gaussian function for space)
    'sigma_r' is a int value, which is a sigma value for G_r(gaussian function for range)
    You should return result image.
    """
    #print("bilateral filter")
    p = int((kernel_size-1)/2) #number of padding
    w = int((kernel_size + 1) / 2 )#use to move index in output image
    center = w - 1 #center

    #get
    new_shape0 = img.shape[0]+2*p
    new_shape1 = img.shape[1]+2*p
    image_output = np.zeros((new_shape0,new_shape1,img.shape[2])) #padding #img.copy() #np.zeros(img.shape)
    image_output[p:new_shape0-p, p:new_shape1-p] = img.copy()
    img = image_output.copy()
    #image_output = img.copy() #np.zeros(image.shape)
    #print("image shape is {} {}".format(img.shape[0], img.shape[1]))
    for x in range(0, new_shape0-kernel_size+1):
        for y in range(0, new_shape1-kernel_size+1):
            for c in range(0, img.shape[2]):
                #print("x, y is {} {}".format(x, y))
                sum = 0
                weights = 0
                tmp = img[x:x+kernel_size, y:y+kernel_size,c]
                #print(tmp)
                for m in range(0, kernel_size):
                    for n in range(0, kernel_size):
                        distance2 = pow(m-center,2) + pow(n-center,2)
                        diff = int(tmp[m,n])-int(tmp[center, center])   ###
                        #diff = int(tmp[m,n])/255-int(tmp[center, center])/255 ###
                        diff_range = pow(diff,2)
                        gaussian_s = np.exp((-0.5)*distance2/pow(sigma_s,2)) #(1/(2*np.pi*pow(sigma_s,2))) multiply do not need.
                        gaussian_r = np.exp((-0.5)*diff_range/pow(sigma_r,2)) #
                        weights += gaussian_s*gaussian_r
                        sum += gaussian_s*gaussian_r*tmp[m,n] #g_s*g_r*intensity of pixel
                image_output[int(x+w-1),int(y+w-1),c] = sum/weights
                #print(image_output)
        #print("x is {}".format(x))
    #img = image_output
    img = image_output[p:new_shape0-p, p:new_shape1-p]
    return img


def apply_my_filter(img, kernel_size): #average filter
    """
    You should implement additional filter using convolution.
    You can use any filters for this function, except median, bilateral filter.
    You can add more arguments for this function if you need.

    You should return result image.
    """
    #print("my filter")

    ## AVerage
    p = int((kernel_size-1)/2) #number of padding
    w = int((kernel_size + 1) / 2 ) #use to move index in output image

    #get mean(average) 
    new_shape0 = img.shape[0]+2*p
    new_shape1 = img.shape[1]+2*p
    image_output = np.zeros((new_shape0,new_shape1,img.shape[2])) #padding #img.copy() #np.zeros(img.shape)
    image_output[p:new_shape0-p, p:new_shape1-p] = img.copy()
    img = image_output.copy()
    for x in range(0, new_shape0-kernel_size+1):
        for y in range(0, new_shape1-kernel_size+1):
            for c in range(0, img.shape[2]):
                #print("x, y is {} {}".format(x, y))
                tmp = img[x:x+kernel_size, y:y+kernel_size,c]
                #print(tmp)
                # t = tmp.flatten() #flatten -> array
                # average = np.mean(t)
                sum = 0
                for m in range(0, kernel_size):
                    for n in range(0, kernel_size):
                        sum += tmp[m,n]
                image_output[int(x+w-1),int(y+w-1),c] = sum / (kernel_size*kernel_size)
                #print(image_output)
    img = image_output[p:new_shape0-p, p:new_shape1-p]
    #print("happy")
    return img


def calculate_rms(img1, img2):
    """
    Calculates RMS error between two images. Two images should have same sizes.
    """
    if (img1.shape[0] != img2.shape[0]) or \
            (img1.shape[1] != img2.shape[1]) or \
            (img1.shape[2] != img2.shape[2]):
        raise Exception("img1 and img2 should have same sizes.")

    diff = np.abs(img1 - img2)
    diff = np.abs(img1.astype(dtype=np.int) - img2.astype(dtype=np.int))
    return np.sqrt(np.mean(diff ** 2))


#main function for test
# ## remove b/f submit
#main function for test
# if __name__ == '__main__':
#     np.random.seed(0)
#     import os
#     folder = "test_images"
#     print(os.getcwd())

#     ## path
#     src_path = os.path.join(os.getcwd(),folder, 'cat_noisy.jpg')
#     clean_path = os.path.join(os.getcwd(),folder, 'cat_clean.jpg')
#     dst_path = os.path.join(os.getcwd(),folder, 'cat_denoised.jpg')    

#     # src_path = os.path.join(os.getcwd(),folder, 'fox_noisy.jpg')
#     # clean_path = os.path.join(os.getcwd(),folder, 'fox_clean.jpg')
#     # dst_path = os.path.join(os.getcwd(),folder, 'fox_denoised.jpg')

    
#     # src_path = os.path.join(os.getcwd(),folder, 'snowman_noisy.jpg')
#     # clean_path = os.path.join(os.getcwd(),folder, 'snowman_clean.jpg')
#     # dst_path = os.path.join(os.getcwd(),folder, 'snowman_denoised.jpg')
    
#     ##task1_2
#     #start = time.time()  # 시작 time 저장
#     task1_2(src_path, clean_path, dst_path)
#     #print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

    
   
    
    