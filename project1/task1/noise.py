import cv2
import numpy as np


## Do not erase or modify any lines already written
## Each noise function should return image with noise

def add_gaussian_noise(image):
    # Use mean of 0, and standard deviation of image itself to generate gaussian noise
    mean = 0
    std = np.std(image) #standard deviation of image
    #print("image is {}".format(image))
    gaussian_noise = np.random.normal(mean, std, image.shape)
    image = image + gaussian_noise #add gaussian_noise to original image
    return image

def add_uniform_noise(image):
    # Generate noise of uniform distribution in range [0, standard deviation of image)
    #print("image is {}".format(type(image)))
    std = np.std(image)
    uniform_noise = np.random.uniform(0, std, image.shape)
    image = image + uniform_noise
    return image

def apply_impulse_noise(image): #color image
    # # Implement pepper noise so that 20% of the image is noisy
    
    row = image.shape[0]
    col = image.shape[1]
    #print("row is {} col is {}".format(row, col))
    num_pixels = row * col
    #print("image shape is {}".format(image.shape)) #427*640
    #print("num_pixels is {}".format(num_pixels))
    num_pepper = int(0.2*num_pixels)
 

    pixel = np.random.choice(num_pixels, num_pepper, replace =False)
    #print(type(pixel))
    #print(pixel)
    #print("pixel is {}".foramt(pixel))
    for i in range(num_pepper):
        x = pixel[i]//col
        y = int(pixel[i]%col)
        image[x][y] = 0 #black 
    #print(len(tlist)) #54656

         

    return image


def rms(img1, img2):
    # This function calculates RMS error between two grayscale images. 
    # Two images should have same sizes.
    if (img1.shape[0] != img2.shape[0]) or (img1.shape[1] != img2.shape[1]):
        raise Exception("img1 and img2 should have the same sizes.")

    diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32))

    return np.sqrt(np.mean(diff ** 2))


if __name__ == '__main__':
    np.random.seed(0)
    original = cv2.imread('bird.jpg', cv2.IMREAD_GRAYSCALE)
    
    gaussian = add_gaussian_noise(original.copy())
    print("RMS for Gaussian noise:", rms(original, gaussian))
    cv2.imwrite('gaussian.jpg', gaussian)
    
    uniform = add_uniform_noise(original.copy())
    print("RMS for Uniform noise:", rms(original, uniform))
    cv2.imwrite('uniform.jpg', uniform)
    
    impulse = apply_impulse_noise(original.copy())
    print("RMS for Impulse noise:", rms(original, impulse))
    cv2.imwrite('impulse.jpg', impulse)