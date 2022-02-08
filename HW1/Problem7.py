from cProfile import label
from pickletools import int4
import numpy as np
from numpy import copy
from PIL import Image
from numpy.linalg import svd as svd
from numpy.linalg import norm as norm
from matplotlib import pyplot as plt
from sys import exit

# Import the image using the Image function in the Pillow package:
img_in = Image.open("UWlogo.png")

print("Inspecting img_in, we see this is a PIL object, not amenable to numpy-style manipulation:")
print(img_in)

# So to manipulate it we will convert the image to a multi-dimensional array (4 matrices)
img=np.asarray(img_in) 

# Print the type and shape of the imported pixel array
print("img has type: ", img.dtype) # (should output uint8, an 8 bit (1 byte) integer)
print("img has shape: ", img.shape) # (should output (761, 1133, 4). Height (pixels), width (pixels), and colors (RBGA), A is 'alpha', transparency between 0 and 1)

# We will use matplotlib's imshow to display the array of pixels as an image, and save to a file HW1_plot00
# fig = plt.figure()
# plt.imshow(img)
# plt.axis('off') # Hide grid lines
# plt.savefig('./HW1_plot00.png',bbox_inches='tight', pad_inches=0)

# Breaking up img into its four matrices:
R = copy(img[:,:,0]) # Red
G = copy(img[:,:,1]) # Green
B = copy(img[:,:,2]) # Blue
A = copy(img[:,:,3]) # Alpha (transparency)

def compute_svd(arr):
    u, sigma, vh = np.linalg.svd(arr, full_matrices=False, compute_uv=True)
    return u, sigma, vh

def compute_diff_norm(x, y):
    return np.linalg.norm(x-y, 'fro')

def compute_matrix_from_svd(u, sigma, vh):
    return np.matmul(np.matmul(u, np.diag(sigma)), vh)

def convert_image_to_grayscale(img):
    M = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return M

def reconstruct_image(u, sigma, vh, rank=100):
    sigma_trim = np.copy(sigma)
    sigma_trim[rank:] = 0
    return compute_matrix_from_svd(u, sigma_trim, vh)

num_channels = 3
labels = ['sigma-red', 'sigma-green', 'sigma-blue']

for i in range(0, num_channels):
    u, sigma, vh = compute_svd(img[:, :, i])
    x = compute_matrix_from_svd(u, sigma, vh)
    norm_diff = compute_diff_norm(x, img[:, :, i])
    print('Norm of difference between image channel {} and reconstructed matrix from its svd = {}'.format(i, norm_diff))
    plt.semilogy(sigma, label=labels[i])
plt.legend()
plt.xlabel('Number of singular values')
plt.ylabel('Log of singular values')
plt.title('Semi-log plot of singular values of the UW logo')
plt.savefig('./Sigma_plot.png', bbox_inches='tight', pad_inches=0.3)

plt.clf()
M = convert_image_to_grayscale(img)
plt.imshow(M, cmap='gray', vmin=0, vmax=255)
plt.axis('off')
plt.savefig('./M_grayscale.png',bbox_inches='tight', pad_inches=0)

u, sigma, vh = compute_svd(M)
print(u.shape, sigma.shape, vh.shape)

ranks = [1, 5, 50, 200]

plt.clf()
fig, axs = plt.subplots(2, 2)
for i in range(len(ranks)):
    M_compressed = reconstruct_image(u, sigma, vh, ranks[i])
    axs[int(i/2), int(i%2)].imshow(M_compressed, cmap='gray', vmin=0, vmax=255)
    axs[int(i/2), int(i%2)].axis('off')
    axs[int(i/2), int(i%2)].set_title('Rank = {}'.format(ranks[i]))
    extent = axs[int(i/2), int(i%2)].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('./M_compressed_Rank_{}.png'.format(ranks[i]), bbox_inches=extent, pad_inches=0)

plt.savefig('./M_compressed_images.png',bbox_inches='tight', pad_inches=0)

# Now let's manipulate the image and save it to a file:
# fig = plt.figure()
# X, Y = np.meshgrid(np.linspace(0,1,1133),np.linspace(0,1,761))
# img[:,:,0] = 0 # no red
# img[:,:,3] = np.uint8(img[:,:,3]*(1-X)**2) # make alpha X-dependent

# plt.imshow(img)
# # Hide grid lines
# plt.axis('off')
# # Save to file
# plt.savefig('./HW1_plot0.png',bbox_inches='tight', pad_inches=0)
