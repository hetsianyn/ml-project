import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

A = imread('flower.jpg')
X = np.mean(A, -1)

B = imread('flower_noise_3.png')
Y = np.mean(B, -1)

img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
# plt.savefig('output/singular_values{0}.png'.format(r))
plt.show()

img = plt.imshow(Y)
img.set_cmap('gray')
plt.axis('off')
# plt.savefig('output/singular_values{0}.png'.format(r))
plt.show()

# Calculate U (u), Σ (s) and V (vh)
U, S, VT = np.linalg.svd(Y,full_matrices=False)
S = np.diag(S)
# Remove sigma values below threshold (250)
print(S)
r = 500
s_cleaned = S[:r,0:r]
print(s_cleaned.shape)
# Calculate A' = U * Σ (cleaned) * V
img_denoised = U[:,:r] @ s_cleaned @ VT[:r,:]

img = plt.imshow(img_denoised)
img.set_cmap('gray')
plt.axis('off')
# plt.savefig('output/singular_values{0}.png'.format(r))
plt.show()

