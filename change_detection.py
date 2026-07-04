import practice as np
import matplotlib.pyplot as plt

np.random.seed(0)
h, w = 50, 50

#healthy forest(T_1)
ndvi_t1 = np.random.uniform(0.5, 0.85, (h, w))

#Time 2: forest cleared in two patches
ndvi_t2 = ndvi_t1.copy()
ndvi_t2[10:20, 10:25] = np.random.uniform(0.05, 0.15, (10, 15))  # patch 1
ndvi_t2[30:40, 30:45] = np.random.uniform(0.05, 0.15, (10, 15))  # patch 2

#change = differnece between both of them
ndvi_diff = ndvi_t2 - ndvi_t1

#Threshold: significant loss = drop > 0.3
deforestation_mask = ndvi_diff < -0.3

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
kw = dict(cmap='RdYlGn', vmin=0, vmax=1)

axes[0].imshow(ndvi_t1, **kw)
axes[0].set_title('NDVI - time 1 (forest intact)')

axes[1].imshow(ndvi_t2, **kw)
axes[1].set_title('NDVI - time 2 (after clearing)')

axes[2].imshow(deforestation_mask, cmap='Reds')
axes[2].set_title('Deforestation Mask')

plt.tight_layout()
plt.savefig('change_detection.png', dpi=150, bbox_inches='tight')
plt.show()

pct = deforestation_mask.mean() * 100
print(f'Forest loss detected: {pct:.1f}% of study area')