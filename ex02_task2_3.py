import matplotlib.pyplot as plt
import numpy as np

plt.close('all')

#%% Task 2 - Mean and Median

def median(listValues):
    median_value = 0.0
    # TODO
    size = len(listValues)
    sorted_values = np.sort(listValues)
    if size % 2 == 0:
        median_value = (sorted_values[int(size/2-1)]+listValues[int(size/2)])/2
    else:
        median_value = sorted_values[int(size/2)] # python is 0-index
    return median_value

# 1D median & mean
values = [19.7, 556.3, 23.2, 27.5, 16.3, 21.0, 27.2, 495.0, 25.3]
print('Median value = {}'.format(median(values)))
print('Mean value = {}'.format(np.mean(values)))


# %% 2D median & mean

def addBorder_mirror(img, br, bc):
    row,col = img.shape
    imgOut = np.zeros((row +2*br,col+2*bc),np.uint8)
    r = 0
    c = 0
    for px in np.nditer(imgOut[:,:], op_flags=["readwrite"]):
        rI = br - r
        cI = bc - c
        if rI < 0:
            rI = r - br 
        if rI >= row:
            rI = row - (rI - row) - 2
        if cI < 0:
            cI = c - bc
        if cI >= col:
            cI = col - (cI - col) - 2
        px[...] = img[rI,cI]
        c+=1
        if c >= imgOut.shape[1]:
            c=0
            r+=1
    return imgOut


def convolution(img,kernel):  
    rows,cols = kernel.shape
    n = float(rows*cols) # check if kernel is odd
    rI,cI = img.shape
    imgOut = np.zeros((rI,cI),np.float32)
    startC = int(cols/2 )
    startR = int(rows/2 )
    imgBorder = addBorder_mirror(img, startR, startC)
    imgBorder.astype(np.float32)
    r = 0
    c = 0
    for pxOut in np.nditer(imgOut[:,:], op_flags =["writeonly"]):
        it = np.nditer([imgBorder[r : r+2*startR+1 , c : c+2*startC+1],kernel[:,:]],
                       flags=["buffered","external_loop"],
                       op_flags =["readonly"], 
                       op_dtypes=["float64","float64"])
        val = 0.0
        for i,k in it:
            val += np.sum(i*k)
        pxOut[...] = val
        c+= 1
        if c >= imgOut.shape[1]:
            c=0
            r+=1
    return imgOut


def medianImg(img,size):
    imgB = addBorder_mirror(img,int(size/2),int(size/2))
    imgOut = np.zeros(img.shape,img.dtype)
    c=0
    r=0
    for pxOut in np.nditer(imgOut[:,:], op_flags =["writeonly"]):
        it = np.nditer(imgB[r : r+size , c : c+size], # +3 to row, +3 to col
                       flags=["buffered","external_loop"],
                       op_flags =["readonly"])
        for x in it:
            # print(x)
            pxOut[...] = median(x)
        c+=1
        if c >= imgOut.shape[1]:
            c=0
            r+=1
    return imgOut

# Define input
chessboard = np.array([[0,0,0,0,0,255,255,255,255,255],[0,0,0,0,0,255,255,255,255,255],
                       [0,0,0,0,0,255,255,255,255,255],[0,0,0,0,0,255,255,255,255,255],
                       [0,0,0,0,0,255,255,255,255,255],[255,255,255,255,255,0,0,0,0,0],
                       [255,255,255,255,255,0,0,0,0,0],[255,255,255,255,255,0,0,0,0,0],
                       [255,255,255,255,255,0,0,0,0,0],[255,255,255,255,255,0,0,0,0,0]])

# Define kernel
kernel_mean = np.ones((3,3)) / np.size(np.ones((3,3)))

# original image
fig, axes = plt.subplots(1,3)
axes[0].imshow(chessboard)
axes[0].set_title('original')
axes[1].imshow(convolution(chessboard, kernel_mean)) # use odd sized kernel
axes[1].set_title('mean')
axes[2].imshow(medianImg(chessboard, 3)) # use odd number neighbours
axes[2].set_title('median')
plt.show()
plt.tight_layout()
plt.savefig('mean_median_original.png')


chessboard_noisy = np.loadtxt("data/chessboard_noisy.txt")

# noisy image
fig, axes = plt.subplots(1,3)
axes[0].imshow(chessboard_noisy)
axes[0].set_title('noisy')
axes[1].imshow(convolution(chessboard_noisy, kernel_mean)) # use odd sized kernel
axes[1].set_title('mean')
axes[2].imshow(medianImg(chessboard_noisy, 3)) # use odd number neighbours
axes[2].set_title('median')
plt.show()
plt.tight_layout()
plt.savefig('mean_median_noisy.png')


#%% Task 3 - Sobel and Laplace

def sobel(img):
    ksx = np.zeros((3,3),np.float32)   
    # TODO
    ksx[0,:] = [-1, 0, 1]
    ksx[1,:] = [-2, 0, 2]
    ksx[2,:] = [-1, 0, 1]
    
    ksy = np.zeros((3,3),np.float32)
    # TODO
    ksy[0,:] = [-1, -2, -1]
    ksy[1,:] = [ 0,  0,  0]
    ksy[2,:] = [ 1,  2,  1]
    
    Sx = convolution(img, ksx)
    Sy = convolution(img, ksy)
    return Sx, Sy 

def laplace(img):
    kl = np.zeros((3,3),np.float32)
    kl[0,:] = np.add([0, 0,  0], [0, 1,  0])
    kl[1,:] = np.add([1, -2, 1], [0, -2, 0])
    kl[2,:] = np.add([0, 0,  0], [0, 1,  0])
    # TODO
    return convolution(img,kl)

input_img = plt.imread('data/old_town.jpg')
Sy_img, Sx_img = sobel(input_img)
fig, axes = plt.subplots(2,2)
axes[0,0].imshow(Sy_img, cmap = plt.get_cmap("gray"))
axes[0,0].set_title('Sx')
axes[0,1].imshow(Sx_img, cmap = plt.get_cmap("gray"))
axes[0,1].set_title('Sy')
axes[1,0].imshow(np.sqrt(np.square(Sy_img)+np.square(Sx_img)),
                 cmap = plt.get_cmap("gray"))
axes[1,0].set_title('Smag')
axes[1,1].imshow(np.arctan2(Sy_img,Sx_img), cmap = plt.get_cmap("gray"))
axes[1,1].set_title('Sdir')
plt.tight_layout()
plt.savefig('sobel_output.png')

L_img = laplace(input_img)
fig, axes = plt.subplots(1,2)
axes[0].imshow(input_img, cmap = plt.get_cmap("gray"))
axes[0].set_title('Original image')
axes[1].imshow(L_img, cmap = plt.get_cmap("gray"))
axes[1].set_title('Laplace output')
plt.tight_layout()
plt.savefig('laplace_output.png')

