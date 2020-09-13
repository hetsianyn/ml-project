from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np

A = imread('flower.jpg')
X = np.mean(A, -1) # Convert RGB to grayscale

img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
plt.title('Original image')
#plt.savefig('output/original_image.png')
plt.show()

U, S, VT = np.linalg.svd(X,full_matrices=True)
S = np.diag(S)

print(U)
print(S)
print(VT)


j = 0
rank = [5, 20, 100]
for r in (5, 20, 100):
    # Construct approximate image
    Xapprox = U[:,:r] @ S[0:r,:r] @ VT[:r,:]
    plt.figure(j+1)
    j += 1
    img = plt.imshow(Xapprox)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('r = ' + str(r))
    #plt.savefig('output/singular_values{0}.png'.format(r))
    plt.show()

plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.ylabel('Singular value')
plt.xlabel('k')
plt.grid()
#plt.savefig('output/singular_values_plot.png')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Singular Values: Cumulative Sum')
plt.ylabel('Cumulative energy')
plt.xlabel('k')
plt.grid()
#plt.savefig('output/cumulative_sum_plot.png')
plt.show()