import cv2
import matplotlib.pyplot as plt
import numpy as np

##### To-do #####

def fftshift(img):
    '''
    This function should shift the spectrum image to the center.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    # print('img is {}'.format(img))
    #t = img #
    row, col = img.shape
    r = round(row / 2) 
    c = round(col / 2) 

    # print("row is {}".format(row))
    # print("col is {}".format(col))
    # print("r is {}".format(r))
    # print("c is {}".format(c))

    m1 = img[0:r, 0:c]
    m2 = img[0:r, c:]
    m3 = img[r:, 0:c]
    m4 = img[r: , c:]

    m2_m1 = np.concatenate((m2,m1), axis = 1)
    m4_m3 = np.concatenate((m4,m3), axis = 1)
    img = np.concatenate((m4_m3,m2_m1), axis = 0)
    # print("m1 is {}".format(m1))
    # print("m2 is {}".format(m2))
    # print("m3 is {}".format(m3))
    # print("m4 is {}".format(m4))
    # print("m2_m1 is {}".format(m2_m1))
    # print("m4_m3 is {}".format(m4_m3))
    # print("m is {}".format(m))

    #check is it correct
    #print((img == np.fft.fftshift(t))) #
    # print('after fftshift is {}'.format(np.fft.fftshift(img)))
    return img

def ifftshift(img):
    '''
    This function should do the reverse of what fftshift function does.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    #t = img #
    row, col = img.shape
    r = round(row / 2) 
    c = round(col / 2) 

    # print("row is {}".format(row))
    # print("col is {}".format(col))
    # print("r is {}".format(r))
    # print("c is {}".format(c))

    m1 = img[0:r, 0:c]
    m2 = img[0:r, c:]
    m3 = img[r:, 0:c]
    m4 = img[r: , c:]

    # print("m1 is {}".format(m1))
    # print("m2 is {}".format(m2))
    # print("m3 is {}".format(m3))
    # print("m4 is {}".format(m4))

    m2_m1 = np.concatenate((m2,m1), axis = 1)
    m4_m3 = np.concatenate((m4,m3), axis = 1)
    img = np.concatenate((m4_m3,m2_m1), axis = 0)

    #check is it correct
    #print((img == np.fft.ifftshift(t))) #

    return img

def fm_spectrum(img):
    '''
    This function should get the frequency magnitude spectrum of the input image.
    Make sure that the spectrum image is shifted to the center using the implemented fftshift function.
    You may have to multiply the resultant spectrum by a certain magnitude in order to display it correctly.
    '''
    #print("fm_spectrum") #
    f_image = np.fft.fft2(img)
    #k = f_image #
    f_shift = fftshift(f_image) 
    #f_shift = f_image
    m_spectrum = 20*np.log(np.abs(f_shift))
    img = m_spectrum
    #print("m_spectrum is {}".format(m_spectrum)) #
    #check fftshift, ifftshift implementation right
    #f_image = ifftshift(img) #
    #print(f_image == k) #
    #print("fm_spectrum") #
    
    return img

def low_pass_filter(img, r=30):
    '''
    This function should return an image that goes through low-pass filter.
    '''
    #create low_pass_filter(ndarray) in frequency domain
    row, col = img.shape
    low_filter_fs = np.zeros((row, col),dtype = 'complex') #dtype=complex <- multiply with image in frequency domain
    #print(low_filter_fs)
    #print(low_filter_fs.size)
    id_center_row = row // 2 #index of center row
    id_center_col = col // 2 #index of center col
    #print(id_center_row, id_center_col)

    for i in range(row):
        for j in range(col): #(i,j) - (id_center_row, id_center_col)
            distance =(i - id_center_row) ** 2 + (j-id_center_col)**2
            d = np.sqrt(distance)
            if (d<=r):
                low_filter_fs[i, j] = 255 #white
    
    f_image = np.fft.fft2(img)
    f_shift = fftshift(f_image)
    f_low = low_filter_fs * f_shift #element-wise multiplication
    f_low_ishift = ifftshift(f_low)
    img = np.fft.ifft2(f_low_ishift) #inverse Fourier Transform
    img = img.real
    #0-255 values
    img -= img.min()
    img = img*255 / img.max()
    img = img.astype(np.uint8)

    return img

def high_pass_filter(img, r=20):
    '''
    This function should return an image that goes through high-pass filter.
    '''
    #create low_pass_filter(ndarray) in frequency domain
    row, col = img.shape
    high_filter_fs = np.ones((row, col),dtype = 'complex') #dtype=complex <- multiply with image in frequency domain
    id_center_row = row // 2 #index of center row
    id_center_col = col // 2 #index of center col
    #print(id_center_row, id_center_col)

    for i in range(row):
        for j in range(col): #(i,j) - (id_center_row, id_center_col)
            distance =(i - id_center_row) ** 2 + (j-id_center_col)**2
            d = np.sqrt(distance)
            if (d<=r):
                high_filter_fs[i, j] = 0 #black
    
    f_image = np.fft.fft2(img)
    f_shift = fftshift(f_image)
    f_low = high_filter_fs * f_shift #element-wise multiplication
    f_low_ishift = ifftshift(f_low)
    img = np.fft.ifft2(f_low_ishift) #inverse Fourier Transform
    img = img.real
    #0-255 values
    img -= img.min()
    img = img*255 / img.max()
    img = img.astype(np.uint8)

    return img

def denoise1(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''
    # #create denoise1 filter
    row, col = img.shape
    denoise1_filter = np.ones((row, col),dtype = 'complex') #dtype=complex <- multiply with image in frequency domain
    #print(denoise1_filter.shape)
    id_center_row = row // 2 #index of center row
    id_center_col = col // 2 #index of center col

    for i in range(row):
        for j in range(col): #tried a lot...haha
            if(50<=np.abs(i-id_center_row)<=60):
                if(50<=np.abs(j-id_center_col)<=60):
                    denoise1_filter[i][j] = 0 #black
            elif(80<=np.abs(i-id_center_row)<=90):
                if(80<=np.abs(j-id_center_col)<=90):
                    denoise1_filter[i][j] = 0 #black
            

    f_image = np.fft.fft2(img)
    f_shift = fftshift(f_image)
    f_low = denoise1_filter * f_shift #element-wise multiplication
    f_low_ishift = ifftshift(f_low)
    img = np.fft.ifft2(f_low_ishift) #inverse Fourier Transform
    img = img.real
    #0-255 values
    img -= img.min()
    img = img*255 / img.max()
    img = img.astype(np.uint8)


    return img

def denoise2(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''
    #create denoise2 filter
    row, col = img.shape
    denoise2_filter = np.ones((row, col),dtype = 'complex') #dtype=complex <- multiply with image in frequency domain
    #print(denoise1_filter.shape)
    id_center_row = row // 2 #index of center row
    id_center_col = col // 2 #index of center col

    for i in range(row):
        for j in range(col): #(i,j): diagonal
            distance =(i - id_center_row) ** 2 + (j-id_center_col)**2
            d = np.sqrt(distance)
            if (26<=d<=28): #26, 28.   27:x
                denoise2_filter[i, j] = 0 #black

    f_image = np.fft.fft2(img)
    f_shift = fftshift(f_image)
    f_low = denoise2_filter * f_shift #element-wise multiplication
    f_low_ishift = ifftshift(f_low)
    img = np.fft.ifft2(f_low_ishift) #inverse Fourier Transform
    img = img.real
    #0-255 values
    img -= img.min()
    img = img*255 / img.max()
    img = img.astype(np.uint8)

    return img

#################

# Extra Credit
def dft2(img):
    '''
    Extra Credit. 
    Implement 2D Discrete Fourier Transform.
    Naive implementation runs in O(N^4).
    '''
    return img

def idft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Discrete Fourier Transform.
    Naive implementation runs in O(N^4). 
    '''
    return img

def fft2(img):
    '''
    Extra Credit. 
    Implement 2D Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
    return img

def ifft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
    return img

if __name__ == '__main__':
    img = cv2.imread('task2_filtering.png', cv2.IMREAD_GRAYSCALE)
    noised1 = cv2.imread('task2_noised1.png', cv2.IMREAD_GRAYSCALE)
    noised2 = cv2.imread('task2_noised2.png', cv2.IMREAD_GRAYSCALE)


    low_passed = low_pass_filter(img)
    high_passed = high_pass_filter(img)
    denoised1 = denoise1(noised1)
    denoised2 = denoise2(noised2)

    # save the filtered/denoised images
    cv2.imwrite('low_passed.png', low_passed)
    cv2.imwrite('high_passed.png', high_passed)
    cv2.imwrite('denoised1.png', denoised1)
    cv2.imwrite('denoised2.png', denoised2)

    # draw the filtered/denoised images
    def drawFigure(loc, img, label):
        plt.subplot(*loc), plt.imshow(img, cmap='gray')
        plt.title(label), plt.xticks([]), plt.yticks([])

    drawFigure((2,7,1), img, 'Original')
    drawFigure((2,7,2), low_passed, 'Low-pass')
    drawFigure((2,7,3), high_passed, 'High-pass')
    drawFigure((2,7,4), noised1, 'Noised')
    drawFigure((2,7,5), denoised1, 'Denoised')
    drawFigure((2,7,6), noised2, 'Noised')
    drawFigure((2,7,7), denoised2, 'Denoised')

    drawFigure((2,7,8), fm_spectrum(img), 'Spectrum')
    drawFigure((2,7,9), fm_spectrum(low_passed), 'Spectrum')
    drawFigure((2,7,10), fm_spectrum(high_passed), 'Spectrum')
    drawFigure((2,7,11), fm_spectrum(noised1), 'Spectrum')
    drawFigure((2,7,12), fm_spectrum(denoised1), 'Spectrum')
    drawFigure((2,7,13), fm_spectrum(noised2), 'Spectrum')
    drawFigure((2,7,14), fm_spectrum(denoised2), 'Spectrum')

    plt.show()