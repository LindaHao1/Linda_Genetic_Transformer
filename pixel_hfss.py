"""
Script to reproduce the pixelated device geometry in ANSYS HFSS using PyAEDT.
Modifications:
1. Units set to Microns (um).
2. Materials are explicitly defined/customized (Custom_Copper, Custom_FR4, etc.).
3. Bugs fixed (Solid metal, Lumped Port Reference, AxisDir).
"""

import numpy as np
from ansys.aedt.core import Hfss
from ansys.aedt.core.generic.constants import Plane
from typing import Tuple, Optional

def generate_design_matrix(rows: int, cols: int) -> np.ndarray:
    """Generates a random design matrix."""
    return np.random.randint(0, 2, size=(rows, cols))

def create_custom_material(hfss, name, properties):
    """
    Helper to create a material if it doesn't exist, or return it if it does.
    Allows full customization of properties (permittivity, conductivity, etc.)
    """
    if not hfss.materials.checkifmaterialexists(name):
        mat = hfss.materials.add_material(name)
        for prop_name, prop_val in properties.items():
            # Set property (e.g. mat.permittivity = 4.4)
            setattr(mat, prop_name, prop_val)
        return name
    return name

def draw_pixelated_device_hfss(
    design_matrix: np.ndarray,
    # --- Geometry Parameters (Values in Microns) ---
    pitch_x: float = 15,
    pitch_y: float = 15,
    pixel_w: float = 18,
    pixel_h: float = 18,
    tl_w: float = 20.0,
    tl_l: float = 40.0,
    # --- Stackup & Materials ---
    ground_thickness: float = 1.04,
    substrate_thickness: float = 10.3,
    metal_thickness: float = 2.9,
    passivation_thickness: float = 4.98, 
    # --- HFSS Control ---
    project_name: Optional[str] = "Pixelated_Project",
    design_name: Optional[str] = "Pixelated_Design_Micron",
    non_graphical: bool = False,
    close_on_exit: bool = False
):
    """Draws the pixelated device in HFSS with Micron units."""
    
    # 1. Initialize HFSS
    hfss = Hfss(
        project=project_name, 
        design=design_name, 
        non_graphical=non_graphical, 
        student_version=True
    )

    if hfss.modeler is None:
        raise RuntimeError("Failed to initialize HFSS Modeler.")

    # 2. SET UNITS TO MICRONS
    # This is the critical step to ensure 15 = 15um, not 15mm.
    hfss.modeler.model_units = "um"

    # 3. CUSTOMIZE MATERIALS
    # Define custom materials with specific properties here.
    # You can change these values (conductivity, permittivity) as needed.
    
    material_metal = create_custom_material(hfss, "LBMetal", {
        "conductivity": "31630000"  # Siemens/m
    })
    
    material_ground = create_custom_material(hfss, "JAMetal", {
        "conductivity": "53710000"  # Siemens/m
    })
    
    material_substrate = create_custom_material(hfss, "LayersCombined", {
        "permittivity": 4.1,
        "dielectric_loss_tangent": 0.0
    })
    
    material_passivation = create_custom_material(hfss, "passivation", {
        "permittivity": 3.3
    })

    rows, cols = design_matrix.shape
    
    # --- Geometry Calculations ---
    grid_width_centers = (cols - 1) * pitch_x
    x0 = -grid_width_centers / 2
    y0 = -(rows - 1) * pitch_y / 2
    
    # 4. Draw Pixels
    metal_objects = []
    
    for r in range(rows):
        for col in range(cols):
            if design_matrix[r, col] == 1:
                cx = x0 + (col * pitch_x)
                cy = y0 + (r * pitch_y)
                
                pixel_name = f"Pixel_{r}_{col}"
                p = hfss.modeler.create_box(
                    position=[cx - pixel_w/2, cy - pixel_h/2, 0],
                    dimensions_list=[pixel_w, pixel_h, metal_thickness],
                    name=pixel_name,
                    matname=material_metal
                )
                metal_objects.append(p)

    # 5. Draw Transmission Lines
    # TL1 (Left)
    tl1_xmax = x0 - (pitch_x / 2)
    tl1_xmin = tl1_xmax - tl_l
    tl1_ymin = -tl_w / 2
    
    tl1_obj = hfss.modeler.create_box(
        position=[tl1_xmin, tl1_ymin, 0],
        dimensions_list=[tl_l, tl_w, metal_thickness],
        name="TL_Left",
        matname=material_metal
    )
    metal_objects.append(tl1_obj)
    
    # TL2 (Right)
    tl2_xmin = x0 + (cols - 1) * pitch_x + (pitch_x / 2)
    tl2_xmax = tl2_xmin + tl_l
    
    tl2_obj = hfss.modeler.create_box(
        position=[tl2_xmin, tl1_ymin, 0],
        dimensions_list=[tl_l, tl_w, metal_thickness],
        name="TL_Right",
        matname=material_metal
    )
    metal_objects.append(tl2_obj)
    
    # Unite Metal
    top_metal = None
    if metal_objects:
        united_name = hfss.modeler.unite(metal_objects)
        hfss.modeler.objects[united_name].name = "Top_Metal"
        top_metal = hfss.modeler.objects["Top_Metal"]
    else:
        print("Warning: No metal pixels or lines generated.")

    # 6. Draw Ground & Substrate
    margin = 2.0
    
    pixel_top = y0 + (rows - 1) * pitch_y + pixel_h/2
    pixel_bot = y0 - pixel_h/2
    max_y = max(pixel_top, tl_w/2)
    min_y = min(pixel_bot, -tl_w/2)
    
    gnd_width = (tl2_xmax - tl1_xmin) + 2 * margin
    gnd_height = (max_y - min_y) + 2 * margin
    gnd_center_x = (tl1_xmin + tl2_xmax) / 2
    gnd_center_y = (max_y + min_y) / 2
    
    gnd_xmin = gnd_center_x - gnd_width / 2
    gnd_ymin = gnd_center_y - gnd_height / 2
    
    gnd_obj = hfss.modeler.create_box(
        position=[gnd_xmin, gnd_ymin, -substrate_thickness - ground_thickness],
        dimensions_list=[gnd_width, gnd_height, ground_thickness],
        name="Ground_Plane",
        matname=material_ground
    )
    
    sub_obj = hfss.modeler.create_box(
        position=[gnd_xmin, gnd_ymin, -substrate_thickness],
        dimensions_list=[gnd_width, gnd_height, substrate_thickness],
        name="Substrate",
        matname=material_substrate
    )
    
    # 7. Passivation
    if top_metal:
        pass_obj = hfss.modeler.create_box(
            position=[gnd_xmin, gnd_ymin, 0],
            dimensions_list=[gnd_width, gnd_height, passivation_thickness],
            name="Top_Passivation",
            matname=material_passivation
        )
        hfss.modeler.subtract(
            blank_list=[pass_obj],
            tool_list=[top_metal],
            keep_originals=True
        )

    # 8. Assign Ports
    # Port 1 (Left)
    port1_sheet = hfss.modeler.create_rectangle(
        orientation=Plane.YZ,
        origin=[tl1_xmin, tl1_ymin, 0],
        sizes=[tl_w, -substrate_thickness], 
        name="Port1_Sheet"
    )
    
    hfss.lumped_port(
        assignment=port1_sheet.name,
        reference=[gnd_obj.name],
        create_port_sheet=False,
        integration_line=hfss.AxisDir.ZNeg,
        impedance=50,
        name="P1",
        renormalize=True
    )

    # Port 2 (Right)
    port2_sheet = hfss.modeler.create_rectangle(
        orientation=Plane.YZ,
        origin=[tl2_xmax, tl1_ymin, 0],
        sizes=[tl_w, -substrate_thickness], 
        name="Port2_Sheet"
    )

    hfss.lumped_port(
        assignment=port2_sheet.name,
        reference=[gnd_obj.name],
        create_port_sheet=False,
        integration_line=hfss.AxisDir.ZNeg,
        impedance=50,
        name="P2",
        renormalize=True
    )

    # 9. Radiation Air Box
    # 50% padding works regardless of unit (applies to bounding box)
    region = hfss.modeler.create_region(pad_percent=50)
    hfss.assign_radiation_boundary_to_objects(assignment=region.name)

    # 10. Analysis Setup
    setup_name = "Setup1"
    setup = hfss.create_setup(name=setup_name)
    setup.props["Frequency"] = "60GHz"
    setup.props["MaxDeltaS"] = 0.02
    setup.props["MaximumPasses"] = 12

    # 11. Frequency Sweep (0.01 - 60 GHz)
    hfss.create_linear_step_sweep(
        setup=setup_name,
        unit="GHz",
        start_frequency=0.01,
        stop_frequency=60,
        step_size=0.5, 
        name="Sweep1",
        save_fields=False,
        sweep_type="Interpolating"
    )

    hfss.save_project()
    print(f"Project saved: {hfss.project_file}")
    
    return hfss

if __name__ == "__main__":
    # rows, cols = 5, 5
    # matrix = generate_design_matrix(rows, cols)
    # read from npy
    matrix = np.load("pixelated_device_20260118_005234_264557.npy")
    print("Design Matrix:")
    print(matrix)
    try:
        hfss = draw_pixelated_device_hfss(
            design_matrix=matrix,
            non_graphical=False,
            design_name="test1"
        )
        print("Done.")
    except Exception as e:
        print(f"Failed to run HFSS integration: {e}")