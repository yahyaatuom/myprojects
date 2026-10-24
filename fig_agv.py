import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch, Circle
from matplotlib.path import Path

# ======================================================================
# ISOMETRIC PROJECTION MATH
# ======================================================================
def iso_proj(x, y, z=0):
    """Projects 3D (x,y,z) to 2D isometric (u,v)."""
    u = (x - y) * np.cos(np.radians(30))
    v = (x + y) * np.sin(np.radians(30)) + z
    return u, v

# ======================================================================
# DRAWING PRIMITIVES
# ======================================================================
def draw_cuboid(ax, x, y, w, d, h, color_top, color_left, color_right, z=0, alpha=1.0):
    """Draws a 3D isometric cuboid (building, car chassis, etc.)"""
    # Top face
    pts_top = [iso_proj(x, y, z+h), iso_proj(x+w, y, z+h), iso_proj(x+w, y+d, z+h), iso_proj(x, y+d, z+h)]
    ax.add_patch(Polygon(pts_top, color=color_top, alpha=alpha, zorder=10))
    # Left face
    pts_left = [iso_proj(x, y, z), iso_proj(x+w, y, z), iso_proj(x+w, y, z+h), iso_proj(x, y, z+h)]
    ax.add_patch(Polygon(pts_left, color=color_left, alpha=alpha, zorder=10))
    # Right face
    pts_right = [iso_proj(x+w, y, z), iso_proj(x+w, y+d, z), iso_proj(x+w, y+d, z+h), iso_proj(x+w, y, z+h)]
    ax.add_patch(Polygon(pts_right, color=color_right, alpha=alpha, zorder=10))

def draw_ground(ax, x, y, w, d, color):
    """Draws flat ground tiles (roads, grass)."""
    pts = [iso_proj(x, y, 0), iso_proj(x+w, y, 0), iso_proj(x+w, y+d, 0), iso_proj(x, y+d, 0)]
    ax.add_patch(Polygon(pts, color=color, zorder=1))

# ======================================================================
# SETUP SCENE
# ======================================================================
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_aspect('equal')
ax.axis('off')
ax.set_facecolor('#f4f7f6') # Light off-white background

# 1. Draw Ground & Roads (Grid)
tile_size = 20
for i in range(5):
    for j in range(5):
        # Draw grass blocks
        draw_ground(ax, i*tile_size, j*tile_size, tile_size*0.9, tile_size*0.9, '#8cc63f')
        # Draw roads (borders)
        draw_ground(ax, i*tile_size + tile_size*0.9, j*tile_size, tile_size*0.1, tile_size, '#4a4a4a')
        draw_ground(ax, i*tile_size, j*tile_size + tile_size*0.9, tile_size, tile_size*0.1, '#4a4a4a')

# 2. Draw Buildings (Cuboids)
building_locs = [(10, 10), (50, 10), (10, 50), (50, 50), (30, 30)]
for bx, by in building_locs:
    draw_cuboid(ax, bx, by, 12, 12, 15, '#5fa8d3', '#2a6f97', '#3a86b5', z=0)

# 3. Draw Central AP Tower
ap_x, ap_y = 30, 30
draw_cuboid(ax, ap_x, ap_y, 4, 4, 2, '#aaaaaa', '#888888', '#999999', z=0) # Base
# Tower top (Lattice)
u, v = iso_proj(ap_x+2, ap_y+2, 25)
ax.plot([u, u], [v-23, v], color='red', lw=4, zorder=11) # Red antenna
ax.scatter(u, v, color='red', s=50, zorder=12)

# ======================================================================
# AGVS AND SIGNALS (Your System Model Variables)
# ======================================================================
# Define AGV positions in 3D space (x, y) and their transmission types
agvs = [
    {'x': 5,  'y': 15, 'type': 'V2I', 'color': '#e63946'}, # Red car
    {'x': 25, 'y': 45, 'type': 'V2V', 'color': '#1d3557'}, # Blue car
    {'x': 45, 'y': 25, 'type': 'V2I', 'color': '#f1faee'}, # White car
    {'x': 15, 'y': 5,  'type': 'V2V', 'color': '#457b9d'}, # Blue car
]

for agv in agvs:
    cx, cy = agv['x'], agv['y']
    # Draw AGV (Small cuboid)
    draw_cuboid(ax, cx, cy, 3, 1.5, 1, agv['color'], agv['color'], agv['color'], z=0)
    
    # Project 3D coordinates to 2D for the signal lines
    u_car, v_car = iso_proj(cx+1.5, cy+0.75, 1)
    
    if agv['type'] == 'V2I':
        # Signal to AP (Yellow Lightning)
        u_ap, v_ap = iso_proj(ap_x+2, ap_y+2, 15)
        # Draw a zigzag line (lightning effect)
        mid_u, mid_v = (u_car + u_ap)/2, (v_car + v_ap)/2
        zig_u = mid_u + 1.5
        zig_v = mid_v - 1.5
        ax.plot([u_car, zig_u, u_ap], [v_car, zig_v, v_ap], color='#ffd700', lw=3, zorder=8)
        
    elif agv['type'] == 'V2V':
        # Signal to another specific AGV (Red Lightning)
        # Let's say it talks to the first AGV in the list
        target = agvs[0]
        u_tgt, v_tgt = iso_proj(target['x']+1.5, target['y']+0.75, 1)
        
        mid_u, mid_v = (u_car + u_tgt)/2, (v_car + v_tgt)/2
        zig_u = mid_u + 1.5
        zig_v = mid_v - 1.5
        ax.plot([u_car, zig_u, u_tgt], [v_car, zig_v, v_tgt], color='#e63946', lw=3, zorder=8)

# ======================================================================
# LEGEND
# ======================================================================
# Legend box top right
ax.text(55, 60, "V to I", color='#ffd700', fontsize=14, fontweight='bold')
ax.text(55, 62, "V to V", color='#e63946', fontsize=14, fontweight='bold')

# Add lightning bolts to legend
ax.plot([52, 54, 53, 55], [60, 60.5, 61, 61.5], color='#ffd700', lw=3)
ax.plot([52, 54, 53, 55], [62, 62.5, 63, 63.5], color='#e63946', lw=3)

# Set limits based on isometric projection
ax.autoscale_view()
plt.tight_layout()
plt.savefig("isometric_warehouse_network.png", dpi=300, bbox_inches="tight")
plt.show()