import gdsfactory as gf
import numpy as np
from gdsfactory.typings import LayerSpec
from layers import LayerMap
import uuid
from typing import Tuple
from datetime import datetime  # <-- added import


gf.gpdk.PDK.activate()

# ==============================================================================
# 2. Matrix Generation
# ==============================================================================
def generate_design_matrix(rows: int, cols: int) -> np.ndarray:
    return np.random.randint(0, 2, size=(rows, cols))

# ==============================================================================
# 3. Drawing Function
# ==============================================================================
def draw_pixelated_device(
    design_matrix: np.ndarray,
    layer_top: LayerSpec,
    layer_ground: LayerSpec,
    layer_top_port: LayerSpec,
    layer_ground_port: LayerSpec,
    pitch_x: float = 15,
    pitch_y: float = 15,
    pixel_w: float = 18,
    pixel_h: float = 18,
    tl_w: float = 20.0,
    tl_l: float = 40.0,
    port_stick_len: float = 5.0, 
) -> Tuple[gf.Component, str]:
    
    # Use timestamp for the component name (format: YYYYMMDD_HHMMSS_milliseconds)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    unique_name = f"pixelated_device_{timestamp}"
    c = gf.Component(unique_name)
    
    rows, cols = design_matrix.shape
    
    # --- Geometry Calculations ---
    grid_width_centers = (cols - 1) * pitch_x
    x0 = -grid_width_centers / 2
    y0 = -(rows - 1) * pitch_y / 2

    # --- Draw Pixels ---
    pixel_rect = gf.components.rectangle(size=(pixel_w, pixel_h), layer=layer_top, centered=True)
    for r in range(rows):
        for col in range(cols):
            if design_matrix[r, col] == 1:
                ref = c << pixel_rect
                ref.move((x0 + (col * pitch_x), y0 + (r * pitch_y)))

    # --- Draw Transmission Lines ---
    tl_rect = gf.components.rectangle(size=(tl_l, tl_w), layer=layer_top, centered=True)
    
    tl1 = c << tl_rect
    tl1.movey(0)
    tl1.xmax = x0 - (pitch_x / 2)
    
    tl2 = c << tl_rect
    tl2.movey(0)
    tl2.xmin = x0 + (cols - 1) * pitch_x + (pitch_x / 2)

    # --- Draw Ground Plane ---
    margin = 2.0
    total_device_width = (tl2.xmax - tl1.xmin)
    total_device_height = max(((rows - 1) * pitch_y + pixel_h), tl_w)

    c << gf.components.rectangle(
        size=(total_device_width + 2*margin, total_device_height + 2*margin), 
        layer=layer_ground, 
        centered=True
    )

    # ==========================================================================
    # 4. ADD PORTS
    # ==========================================================================
    marker_shape_top = gf.components.rectangle(
        size=(port_stick_len, tl_w), 
        centered=True,
        layer=layer_top_port, 
    )

    marker_shape_gnd = gf.components.rectangle(
        size=(port_stick_len, tl_w), 
        centered=True,
        layer=layer_ground_port, 
    )

    # --- LEFT SIDE (Input) ---
    # Port 1: Top Signal
    p1 = c << marker_shape_top
    p1.xmin = tl1.xmin  # Flush Left
    p1.y = 0            
    
    # Port 2: Ground Reference
    p2 = c << marker_shape_gnd
    p2.xmin = tl1.xmin  # Flush Left
    p2.y = 0

    # --- RIGHT SIDE (Output) ---
    # Port 3: Top Signal
    p3 = c << marker_shape_top
    p3.xmax = tl2.xmax  # Flush Right
    p3.y = 0

    # Port 4: Ground Reference
    p4 = c << marker_shape_gnd
    p4.xmax = tl2.xmax  # Flush Right
    p4.y = 0

    # ==========================================================================
    # 5. RESTORE LABELS
    # ==========================================================================
    # We use tl1.xmin and tl2.xmax to place text exactly at the port edges.
    
    # Left Labels
    c.add_label(text="P1", position=(tl1.xmin, 0), layer=layer_top)
    c.add_label(text="P1_G", position=(tl1.xmin, 0), layer=layer_ground) 

    # Right Labels
    c.add_label(text="P2", position=(tl2.xmax, 0), layer=layer_top)
    c.add_label(text="P2_G", position=(tl2.xmax, -0), layer=layer_ground)

    c.flatten()
    # c2.name = unique_name # Ensure the flattened copy keeps the unique name
    
    return c, unique_name