import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.close('all')

# %% 1. Sampling and coord. systems

def subset(I, x, y, nu, nv):
    # TODO
    p_ul = (x,y) # tuple with img coord. upper left corner
    p_lr = (x+nu,y+nv) # tuple with img coord. lower right corner
    
    S = I[p_ul[1] : p_lr[1], p_ul[0] : p_lr[0] , :] # subset S with all channels (:)
    return S

def uv2world(easting, northing, scale, u, v): # to generate new coordinates
    # TODO
    d_est =  u*scale # offset of the given coord. u and the upper left corner in meter
    d_north = v*scale # offset of the given coord. v and the upper left corner in meter
    u_est = easting + d_est # easting of the given coord. u 
    v_north = northing + d_north # northing of the given coord. u 
    return u_est, v_north


def loadWordFile(path_in):
  # read data as string and convert them to np array with float values
  f = open(path_in, 'r') # 'r' = read
  data_string = f.read()
  data_array = np.fromstring(data_string, dtype=float , sep = '\n')
  f.close()
  scale = data_array[0] # pixel resolution (x)
  scale_negative = data_array[3] # pixel resolution (y)
  easting = data_array[4] # in UTM coordinates (x)
  northing = data_array[5] # in UTM coordinates (y)
  return easting, northing, scale, scale_negative

# path to files
path_tfw = 'data/32692_5336.tfw' # world file
path_I = 'data/32692_5336.tif' # image file
easting, northing, scale, scale_negative = loadWordFile(path_tfw)

# image coord. of the first pixel of the subset S according to the image I
ps_img = np.array([900, 1000])
# width and height of S
nu, nv = [1200, 800]

# image coord. of the given point according to S
p1_img_subset = np.array([1050, 229])
p2_img_subset = np.array([72, 387])
p3_img_subset = np.array([493, 633]) # TODO


img_dop  = plt.imread(path_I) # TODO #input orthophoto image
p1_img = ps_img + p1_img_subset # TODO
p2_img = ps_img + p2_img_subset # TODO
p3_img = ps_img + p3_img_subset # TODO

# plot image
fig, ax = plt.subplots()
ax.plot(p1_img[0], p1_img[1], marker='x', color="yellow", markersize=20)
plt.text(p1_img[0]+100, p1_img[1], 'P1', fontsize=12, color="yellow")
ax.plot(p2_img[0], p2_img[1], marker='x', color="blue", markersize=20)
plt.text(p2_img[0]+100, p2_img[1], 'P2', fontsize=12, color="blue")
ax.plot(p3_img[0], p3_img[1], marker='x', color="red", markersize=20)
plt.text(p3_img[0]+100, p3_img[1], 'P3', fontsize=12, color="red")
ax.imshow(img_dop)
rect = patches.Rectangle((ps_img[0], ps_img[1]), nu, nv, linewidth=3, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.show()
plt.savefig('full_img.png')

img_dop_subset = subset(img_dop, ps_img[0], ps_img[1], nu, nv) # TODO #subset of the original orthophoto image

# world coordinates
ps_world = np.array([2500, 2500]) # TODO; coordinate of the lower right corner
ps_img_world = uv2world(easting, northing, scale, ps_img[0]-ps_world[0], ps_world[1]-ps_img[1]) #; outputs u_est, v_north
p1_img_world = uv2world(easting, northing, scale, p1_img[0]-ps_world[0], ps_world[1]-p1_img[1]) # swap sign due to +/- dir
p2_img_world = uv2world(easting, northing, scale, p2_img[0]-ps_world[0], ps_world[1]-p2_img[1]) # TODO
p3_img_world = uv2world(easting, northing, scale, p3_img[0]-ps_world[0], ps_world[1]-p3_img[1]) # TODO

print(ps_img_world, p1_img_world, p2_img_world, p3_img_world)

# plot subset
fig, ax = plt.subplots()
ax.plot(p1_img_subset[0], p1_img_subset[1], marker='x', color="yellow", markeredgewidth=4, markersize=20)
plt.text(p1_img_subset[0]+50, p1_img_subset[1], 'P1', fontsize=12, color="yellow")
ax.plot(p2_img_subset[0], p2_img_subset[1], marker='x', color="blue", markeredgewidth=4, markersize=20)
plt.text(p2_img_subset[0]+50, p2_img_subset[1], 'P2', fontsize=12, color="blue")
ax.plot(p3_img_subset[0], p3_img_subset[1], marker='x', color="red", markeredgewidth=4, markersize=20)
plt.text(p3_img_subset[0]+50, p3_img_subset[1], 'P3', fontsize=12, color="red")
ax.imshow(img_dop_subset)
plt.show()
plt.savefig('subset_img.png')

