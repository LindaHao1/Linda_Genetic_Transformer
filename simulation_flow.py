from simulation_engine.pixel_gds import generate_design_matrix, draw_pixelated_device
import gdsfactory as gf
import numpy as np
from gdsfactory.typings import LayerSpec
from simulation_engine.layers import LayerMap
import uuid

from simulation_engine.emx_runner import EMXSimInterface
from simulation_engine.snp_processing import process_snp_file
import os

# Active a dummy PDK as this is required by gdsfactory
gf.gpdk.PDK.activate()

# 1. Instantiate your layer map
layermap = LayerMap

# Parameters

rows = 21   # Must be odd to ensure center alignment with transmission line can change in the future
cols = 30
pitch_x: float = 30
pitch_y: float = 30
pixel_w: float = 34
pixel_h: float = 34
tl_w: float = 34.0   
tl_l: float = 60.0
port_stick_len: float = 5.0

def check_connectivity(matrix, rows, cols, pitch_y, pixel_h, tline_width, pitch_x_req):
    from collections import deque
    # Calculate geometric Y-centers for rows to check tline overlap
    y_coords = np.arange(rows) * pitch_y
    center_y = (y_coords[-1] + y_coords[0]) / 2.0
    
    tline_min = center_y - tline_width / 2.0
    tline_max = center_y + tline_width / 2.0
    
    start_nodes = []
    end_nodes = []
    
    # Identify ports: pixels intersecting tline by at least pitch_x
    for r in range(rows):
        p_min = y_coords[r] - pixel_h / 2.0
        p_max = y_coords[r] + pixel_h / 2.0
        
        overlap_len = max(0, min(tline_max, p_max) - max(tline_min, p_min))
        
        if overlap_len >= pitch_x_req - 1e-9:
            if matrix[r, 0] == 1:
                start_nodes.append((r, 0))
            if matrix[r, cols - 1] == 1:
                end_nodes.append((r, cols - 1))
                
    if not start_nodes or not end_nodes:
        return False
        
    # BFS Pathfinding (8-way connectivity)
    queue = deque(start_nodes)
    visited = set(start_nodes)
    targets = set(end_nodes)
    
    while queue:
        r, c = queue.popleft()
        if (r, c) in targets:
            return True
        
        # 8 neighbors (diagonal included)
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]
                     
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if matrix[nr, nc] == 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return False

connectivity = False
# 2. Generate the design matrix independently
while not connectivity:
    my_matrix = generate_design_matrix(rows=rows, cols=cols)
    # Check Connectivity
    connectivity = check_connectivity(my_matrix, rows, cols, pitch_y, pixel_h, tl_w, pitch_x)
    if not connectivity:
        print("Matrix did not satisfy connectivity/intersection check. Regenerating...")

# 3. Pass the matrix and the specific layers into the drawing function
device, name = draw_pixelated_device(
    design_matrix=my_matrix,
    layer_top=layermap.LB,
    layer_ground=layermap.JA,
    layer_top_port=layermap.LB_PIN,
    layer_ground_port=layermap.JA_PIN,
    tl_w=tl_w, 
    pitch_x=pitch_x,
    pitch_y=pitch_y,
    pixel_w=pixel_w,
    pixel_h=pixel_h,
    tl_l=tl_l,
    port_stick_len=port_stick_len
)
device.write_gds(f"demo_gds/{name}.gds")  # Write it to a GDS file. You can open it in klayout.

# Plot the matrix in black and white and save as numpy file
import matplotlib.pyplot as plt
plt.imshow(my_matrix, cmap='gray_r', interpolation='nearest')
plt.title("Generated Design Matrix")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.savefig(f"demo_gds/{name}_binary.png")
np.save(f"demo_gds/{name}.npy", my_matrix)

fig = device.plot(return_fig=True)
fig.savefig(f"demo_gds/{name}.png")
print(f"Generated GDS and PNG for device: {name} at demo_gds/{name}.gds")

# Start simulation
# ==============================================================================

# 1. Define where you want to save the results
output_folder = "./simulation_results_batch_1"

# 2. Instantiate the interface
runner = EMXSimInterface(destination_folder=output_folder)

# 3. Define your inputs
my_gds_list = [
    f"demo_gds/{name}.gds",
    # "another_device.gds",
]

my_cell_names = [
    name,
    # "another_device_top",
]

# 4. Run Batch
s_files = runner.run_batch(my_gds_list, my_cell_names)

print("\n--- Batch Complete ---")
print("Generated S-parameter files:")
for idx, f in enumerate(s_files):
    print(f)
    # 5. Process each S-parameter file
    path = os.path.join(output_folder, my_cell_names[idx])
    os.makedirs(path, exist_ok=True) 
    process_snp_file(f, save_path=path)