#Compute NDVI - NumPy operates element-wise on entire arrays
#np.where avoids division by zero
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
height, width = 50, 50

nir = np.random.uniform(0.4,0.8, (height, width))
red = np.random.uniform(0.05, 0.2, (height, width))

nir[20:30, 20:30] = np.random.uniform(0.1, 0.2, (10, 10))  # low NIR
red[20:30, 20:30] = np.random.uniform(0.3, 0.5, (10, 10))  # high Red


ndvi = np.where(
    (nir+red) ==0,
    0,
    (nir - red) / (nir + red)
)

print('NDVI stats:')
print(f' Min:   {ndvi.min():.3f}')
print(f'    Max:  {ndvi.max():.3f}')
print(f'    Mean: {ndvi.mean():.3f}')

#visualize
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im1 = axes[0].imshow(ndvi,  cmap ='RdYlGn', vmin =-1, vmax=1)
axes[0].set_title('NDVI map')
plt.colorbar(im1, ax=axes[0], label='NDVI')

#Binary: forest vs deforested
forest_mask = ndvi > 0.3
im2 = axes[1].imshow(forest_mask, cmap='Greens')
axes[1].set_title('Forest Mask (NDVI > 0.3)')

plt.tight_layout()
plt.savefig('ndvi_test.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: ndvi_test.png')