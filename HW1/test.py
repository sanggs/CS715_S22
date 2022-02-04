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
fig = plt.figure()
plt.imshow(img)
plt.axis('off') # Hide grid lines
plt.savefig('./HW1_plot00.png',bbox_inches='tight', pad_inches=0)

# Breaking up img into its four matrices:
R = copy(img[:,:,0]) # Red
G = copy(img[:,:,1]) # Green
B = copy(img[:,:,2]) # Blue
A = copy(img[:,:,3]) # Alpha (transparency)

# Now let's manipulate the image and save it to a file:
fig = plt.figure()
X, Y = np.meshgrid(np.linspace(0,1,1133),np.linspace(0,1,761))
img[:,:,0] = 0 # no red
img[:,:,3] = np.uint8(img[:,:,3]*(1-X)**2) # make alpha X-dependent

plt.imshow(img)
# Hide grid lines
plt.axis('off')
# Save to file
plt.savefig('./HW1_plot0.png',bbox_inches='tight', pad_inches=0)
