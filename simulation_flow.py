from pixel_gds import generate_design_matrix, draw_pixelated_device
import gdsfactory as gf
import numpy as np
from gdsfactory.typings import LayerSpec
from layers import LayerMap
import uuid

from emx_runner import EMXSimInterface
from snp_processing import process_snp_file
import os

# Active a dummy PDK as this is required by gdsfactory
gf.gpdk.PDK.activate()

# 1. Instantiate your layer map
layermap = LayerMap

# 2. Generate the design matrix independently
my_matrix = generate_design_matrix(rows=20, cols=30)

# 3. Pass the matrix and the specific layers into the drawing function
device, name = draw_pixelated_device(
    design_matrix=my_matrix,
    layer_top=layermap.LB,
    layer_ground=layermap.JA,
    layer_top_port=layermap.LB_PIN,
    layer_ground_port=layermap.JA_PIN,
    tl_w=20.0 # Thin transmission line
)
device.write_gds(f"demo_gds/{name}.gds")  # Write it to a GDS file. You can open it in klayout.

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