import numpy as np
import os
import sys
import glob
import cv2

percentage = float(sys.argv[1])
#print("percentage is {}".format(percentage))

## read train image file list first
# print(os.getcwd())
train_path = train_path = os.path.join(os.getcwd(), 'faces_training')

train_images = sorted(glob.glob(train_path+'/*.pgm'))
# print(train_images)
n = len(train_images)

# make student ID directory
foldername = "2019147531"
if(os.path.isdir(foldername)==False):
    os.mkdir(foldername)

# make matrix of dataset
orig_M_data = np.zeros((32256,39))
M_data = np.zeros((32256,39))

for idx,file in enumerate(train_images):   
    # read image file with cv2
    img = cv2.imread(train_images[idx], cv2.IMREAD_GRAYSCALE)
    w, h = img.shape
    Dimension = w*h
    # print(Dimension)
    vec_img = img.reshape(w*h, 1) #32256 dimension. a column vector
    #print(vec_img.shape)

    # refill M_data
    orig_M_data[:, idx:idx+1] = vec_img
    #print(orig_M_data)

# center M_data
#M_x = np.zeros((32256,1))
M_x = np.zeros((32256,39)) #(D,N)
#print(M_x.shape)
for i in range(32256):
    mean = np.mean(orig_M_data[i:i+1, :])
    #mean = np.mean(M_data[:, i:i+1])
    #print(mean)
    M_x[i, :] = mean #a row vector
# print(M_x)
# compute std
#std_x = np.zeros((n,1))
# std_x = np.zeros((32256,1))
# for i in range(32256):
#     #std = np.std(M_data[:, i:i+1])
#     std = np.std(orig_M_data[i:i+1, :])
#     #print(mean)
#     std_x[i, :] = std # a row vector

# Normalizing M_data
for i in range(32256):
    #print("i is {}".format(i))
    for j in range(39):
        #print("M_X is {} std_x is {}".format(M_x[i, :], std_x[i, :]))
        M_data[i, j] = (orig_M_data[i, j] - M_x[i, j]) #/ std_x[i, 0]


# # Run SVD 
#U, S, V = np.linalg.svd(orig_M_data,full_matrices=False)
U, S, V = np.linalg.svd(M_data,full_matrices=False) ######### normalize
# print(U.shape) # 96768,96768 -> 96768,39 (full_matrices=False)
# print(S.shape) # 96768, 39 -> 39, (full_matrices=False)
# print(V.shape) #39,39
#print(S)
eigenvalues = pow(S,2)
eigenvalues_norm = eigenvalues/np.sum(eigenvalues)
# print(eigenvalues_norm)
# snormsum = 0
# for i in range(27):
#     snormsum += S_norm[i]
# print(snormsum)
# print("0-32 -> sum is {}".format(snormsum))
# snormsum += S_norm[33]
# print("0-33 -> sum is {}".format(snormsum))
# print(S_norm)
# print(sum(S_norm))

# choose d (for dimensionality reduction)
sum = 0
for i in range(len(eigenvalues_norm)):
    sum += eigenvalues_norm[i]
    #print(sum)
    if (sum >= percentage):
        d = i+1
        break
#print(d)

# Output.txt step1 
outputpath =os.path.join(os.getcwd(), '2019147531','output.txt')
with open(outputpath, "a") as f:
    print("########## STEP 1 ##########", file=f) 
    #f.write("########## STEP 1 ########## ")
    str_percentage = str(percentage)
    print("Input Percentage: "+str_percentage, file=f) 
    #f.write("Input Percentage: "+str_percentage)
    str_d = str(d)
    print("Selected Dimension: "+str_d, file=f) 
    #f.write("Selected Dimension: "+str_d)
    f.close()

# using eigenvectors (0~d-1)
# Image Reconstruction
error_list = []
new_basis = U[:, :d] #(D, d) #column = eigenface
new_coefficient = np.matmul(np.transpose(new_basis), M_data) #(d, N) = (d,D)*(D,N)
new_surface = np.matmul(new_basis, new_coefficient) #(D,N)
matrix_re_dataset = new_surface + M_x 

for i in range(39):
    img_name = os.path.basename(train_images[i]) 
    dst_path = os.path.join(os.getcwd(), '2019147531', img_name)
    # new_coefficient = np.matmul(np.transpose(new_basis), M_data[:, i]) ######## normalize
    # new_surface = np.matmul(new_basis, new_coefficient)
    # vec_re_image = new_surface + M_x[:, i] ########## normalize M_x[:, 0]
    vec_re_image = matrix_re_dataset[:, i]
    # print(vec_re_image.shape)
    difference = orig_M_data[:, i]-vec_re_image
    # print("difference.shape")
    # print(difference.shape)
    reconstructed_error = 0
    for j in range(32256):
        reconstructed_error += pow(difference[j],2)
    reconstructed_error = reconstructed_error/32256
    error_list.append(reconstructed_error)
    re_img = vec_re_image.reshape(w,h) 
    cv2.imwrite(dst_path, re_img) # 

# #print(type(error_list))
average_error = np.sum(error_list)/39

# Output.txt step2
# remove # before submit
outputpath =os.path.join(os.getcwd(), '2019147531','output.txt')
with open(outputpath, "a") as f:
    print(file = f)
    print("########## STEP 2 ##########", file=f) 
    print("Reconstruction error", file=f) 
    print("Average : {0:0.4f}".format(average_error), file=f) 
    for i in range(39):
        print("{0:02d}: {1:0.4f}".format(i+1, error_list[i]), file=f)
    #f.write("Selected Dimension: "+str_d)
    f.close()



# step3
test_path  = os.path.join(os.getcwd(), 'faces_test')

test_images = sorted(glob.glob(test_path+'/*.pgm'))
n_test = len(test_images)
# print(test_images)
# make matrix of dataset
test_orig_M_data = np.zeros((32256,n_test))
test_M_data = np.zeros((32256,n_test))


for idx,file in enumerate(test_images):   
    # read image file with cv2
    img = cv2.imread(test_images[idx], cv2.IMREAD_GRAYSCALE)
    w, h = img.shape
    Dimension = w*h
    # print(Dimension)
    vec_img = img.reshape(w*h, 1) #32256 dimension. a column vector
    #print(vec_img.shape)
    # refill M_data
    test_orig_M_data[:, idx:idx+1] = vec_img
    #print(orig_M_data)

for i in range(32256):
    #print("i is {}".format(i))
    for j in range(n_test):
        #print("M_X is {} std_x is {}".format(M_x[i, :], std_x[i, :]))
        test_M_data[i, j] = (test_orig_M_data[i, j] - M_x[i, j]) #/ std_x[i, 0]
test_new_coefficient = np.matmul(np.transpose(new_basis), test_M_data) #(d, 5) = (d,D)*(D,5)

test_train_match = []
for i in range(n_test):
    query_weight = test_new_coefficient[:, i:i+1] #column (d,1)
    # print(query_weight.shape)
    Weights = new_coefficient #(d, N) = (d,D)*(D,N)
    # print(Weights.shape)
    l2_distance = np.linalg.norm((Weights-query_weight), axis = 0)
    argmin = np.argmin(l2_distance) #index 
    test_train_match.append(argmin)
    # print(argmin)

# Output.txt step3
# remove # before submit
outputpath =os.path.join(os.getcwd(), '2019147531','output.txt')
with open(outputpath, "a") as f:
    print(file = f)
    print("########## STEP 3 ##########", file=f) 
    for i in range(n_test):
        # print(i)
        test_img_name = os.path.basename(test_images[i]) 
        train_img_name = os.path.basename(train_images[test_train_match[i]]) 
        print(str(test_img_name)+" ==> "+str(train_img_name), file = f)
    
    f.close()