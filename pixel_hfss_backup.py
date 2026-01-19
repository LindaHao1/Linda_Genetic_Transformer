"""

Script to reproduce the pixelated device geometry in ANSYS HFSS using PyAEDT.

"""



import numpy as np

from ansys.aedt.core import Hfss

from ansys.aedt.core.generic.constants import Plane

from typing import Tuple, Optional



def generate_design_matrix(rows: int, cols: int) -> np.ndarray:

    """Generates a random design matrix (same as pixel_gds.py)."""

    return np.random.randint(0, 2, size=(rows, cols))



def draw_pixelated_device_hfss(

    design_matrix: np.ndarray,

    # --- Geometry Parameters (Match GDS) ---

    pitch_x: float = 15,

    pitch_y: float = 15,

    pixel_w: float = 18,

    pixel_h: float = 18,

    tl_w: float = 20.0,

    tl_l: float = 40.0,

    # --- Stackup & Materials ---

    ground_thickness: float = 2.0,

    substrate_thickness: float = 100.0,

    metal_thickness: float = 2.0,

    passivation_thickness: float = 4.0, # Height of protection layer (must be > metal_thickness if encapsulating)

    material_ground: str = "copper",

    material_substrate: str = "FR4_epoxy",

    material_metal: str = "copper",

    material_passivation: str = "Silicon_dioxide",

    # --- HFSS Control ---

    project_name: Optional[str] = None,

    design_name: Optional[str] = None,

    non_graphical: bool = False,

    close_on_exit: bool = False

):

    """

    Draws the pixelated device in HFSS.

   

    Stackup (Z-axis):

    - Ground Plane: z = [-substrate_th - ground_th, -substrate_th]

    - Substrate:    z = [-substrate_th, 0]

    - Metal (Pixels/TLines): z = [0, metal_th]

    - Passivation (Top Material): z = [0, passivation_th]

      * Note: Passivation will be (Box - Metal) as requested.

    """

   

    # Initialize HFSS

    hfss = Hfss(projectname=project_name, designname=design_name, non_graphical=non_graphical)

   

    # Define Dimensions and Variables if desired, but here we draw directly for simplicity

    # or we can use the values provided.

   

    rows, cols = design_matrix.shape

   

    # --- Geometry Calculations (matching pixel_gds.py) ---

    # Center logic

    grid_width_centers = (cols - 1) * pitch_x

    x0 = -grid_width_centers / 2

    y0 = -(rows - 1) * pitch_y / 2

   

    # 1. Draw Pixels

    metal_objects = []

   

    pixel_x_start = x0

    pixel_y_start = y0

   

    # We create one pixel and duplicate/translate or create multiple.

    # Creating multiple is easier for scripting without dealing with duplication commands complexities in PyAEDT sometimes.

   

    for r in range(rows):

        for col in range(cols):

            if design_matrix[r, col] == 1:

                cx = x0 + (col * pitch_x)

                cy = y0 + (r * pitch_y)

               

                # Rectangle in GDS is purely 2D (center, width, height).

                # In HFSS draw_box_center takes [x, y, z] and [x_size, y_size, z_size]

                # Note: PyAEDT draw_box usually takes corner [x, y, z] and sizes.

                # Let's use draw_box with center calculation.

               

                x_min = cx - pixel_w / 2

                y_min = cy - pixel_h / 2

               

                pixel_name = f"Pixel_{r}_{col}"

                p = hfss.modeler.create_box(

                    position=[x_min, y_min, 0],

                    dimensions_list=[pixel_w, pixel_h, metal_thickness],

                    name=pixel_name,

                    matname=material_metal

                )

                metal_objects.append(p)



    # 2. Draw Transmission Lines

    # TL1 (Left)

    # alignment logic from GDS:

    # tl1.movey(0) -> centered at y=0

    # tl1.xmax = x0 - (pitch_x / 2)

    # tl_rect size=(tl_l, tl_w)

   

    # TL1 geometry

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

    # tl2.xmin = x0 + (cols - 1) * pitch_x + (pitch_x / 2)

    tl2_xmin = x0 + (cols - 1) * pitch_x + (pitch_x / 2)

    tl2_ymin = -tl_w / 2

   

    tl2_obj = hfss.modeler.create_box(

        position=[tl2_xmin, tl2_ymin, 0],

        dimensions_list=[tl_l, tl_w, metal_thickness],

        name="TL_Right",

        matname=material_metal

    )

    metal_objects.append(tl2_obj)

   

    # Unite all metal objects into one "Top_Metal" object

    if metal_objects:

        hfss.modeler.unite(metal_objects)

        # Rename the resulting object (usually takes the name of the first or last)

        # Let's find the surviving object name

        # PyAEDT unite returns the boolean result name or object

        top_metal_name = metal_objects[0].name

        hfss.modeler.create_object_from_face(hfss.modeler.get_object_faces(top_metal_name)) # ensure it's solid? It is already a box.

        # Rename for clarity

        hfss.modeler.objects[top_metal_name].name = "Top_Metal"

        top_metal = hfss.modeler.objects["Top_Metal"]

    else:

        print("Warning: No metal pixels or lines generated.")

        top_metal = None



    # 3. Draw Ground Plane

    # Dimensions from GDS logic:

    # margin = 2.0

    # total_device_width = (tl2.xmax - tl1.xmin) -> (tl2_xmin + tl_l) - tl1_xmin

    # total_device_height = max(((rows - 1) * pitch_y + pixel_h), tl_w)

   

    margin = 2.0

    tl2_xmax = tl2_xmin + tl_l

    total_device_width = tl2_xmax - tl1_xmin

    # pixel_array_height calculation for margin:

    # centers range from y0 to y0 + (rows-1)*pitch.

    # top edge = y0 + (rows-1)*pitch + pixel_h/2

    # bottom edge = y0 - pixel_h/2

    # height spans = (rows-1)*pitch + pixel_h

    # tl_w is centered at 0.

   

    pixel_top_edge = y0 + (rows - 1) * pitch_y + pixel_h/2

    pixel_bottom_edge = y0 - pixel_h/2

   

    max_y = max(pixel_top_edge, tl_w/2)

    min_y = min(pixel_bottom_edge, -tl_w/2)

   

    total_device_height = max_y - min_y

   

    gnd_width = total_device_width + 2 * margin

    gnd_height = total_device_height + 2 * margin

   

    # Center of ground plane

    # Midpoint of X: (tl1_xmin + tl2_xmax) / 2

    gnd_center_x = (tl1_xmin + tl2_xmax) / 2

    # Midpoint of Y: (max_y + min_y) / 2

    gnd_center_y = (max_y + min_y) / 2

   

    gnd_xmin = gnd_center_x - gnd_width / 2

    gnd_ymin = gnd_center_y - gnd_height / 2

    gnd_z = -substrate_thickness - ground_thickness

   

    gnd_obj = hfss.modeler.create_box(

        position=[gnd_xmin, gnd_ymin, gnd_z],

        dimensions_list=[gnd_width, gnd_height, ground_thickness],

        name="Ground_Plane",

        matname=material_ground

    )

   

    # 4. Draw Substrate

    sub_z = -substrate_thickness

    sub_obj = hfss.modeler.create_box(

        position=[gnd_xmin, gnd_ymin, sub_z],

        dimensions_list=[gnd_width, gnd_height, substrate_thickness],

        name="Substrate",

        matname=material_substrate

    )

   

    # 5. Draw Top Material (Passivation)

    # "It should be a box minus all the pixels and tlines"

    # We draw a box covering the same area as substrate? Or covering the metal?

    # Usually passivation covers the chip surface. Let's use the ground/substrate footprint.

   

    if top_metal:

        pass_z = 0

        pass_obj = hfss.modeler.create_box(

            position=[gnd_xmin, gnd_ymin, pass_z],

            dimensions_list=[gnd_width, gnd_height, passivation_thickness],

            name="Top_Passivation",

            matname=material_passivation

        )

       

        # Boolean Subtract: Top_Passivation = Top_Passivation - Top_Metal

        # We need to clone Top_Metal first because subtract consumes the tool object by default unless keep_originals is True.

        # The user said: "box minus all the pixels" -> The result is the box with holes.

        # But we still need the metal there physically.

        # So we must KEEP the metal tool.

       

        hfss.modeler.subtract(

            blank_list=[pass_obj],

            tool_list=[top_metal],

            keep_originals=True

        )



    # 6. Assign Excitation (Ports)

    # Adding lumped ports at the ends of the transmission lines.

    # Port 1 (Left) -> between TL_Left edge and Ground

    # Port 2 (Right) -> between TL_Right edge and Ground

   

    # We need to create a sheet for the port.

    # Left Port:

    # x = tl1_xmin

    # y range: tl1_ymin to tl1_ymin + tl_w

    # z range: 0 to -substrate_thickness (if simplified port) or

    # usually from signal trace down to ground.

   

    # Create Box/Rect for Port 1

    port1_sheet = hfss.modeler.create_rectangle(

        orientation=Plane.YZ,

        origin=[tl1_xmin, tl1_ymin, 0],

        sizes=[tl_w, -substrate_thickness],

        name="Port1_Sheet"

    )

    # Using integration line from bottom (ground) to top (trace) i.e. z = -sub to z=0

    # Create Lumped Port

    hfss.create_lumped_port(

        input_object=port1_sheet.name,

        axisdir="-Z", # vector from signal to ground ? Or Ground to Signal?

                      # Usually integration line is drawn. PyAEDT helper might auto-assign.

                      # Let's specify explicitly if needed. axisdir refers to integration line direction.

                      # from z=0 to z=-sub is -Z.

        portname="P1",

        renorm=True,

        renorm_impedance=50

    )



    # Create Box/Rect for Port 2

    # x = tl2_xmax

    port2_sheet = hfss.modeler.create_rectangle(

        orientation=Plane.YZ,

        origin=[tl2_xmax, tl1_ymin, 0], # tl1_ymin same as tl2_ymin (tl_w centered)

        sizes=[tl_w, -substrate_thickness],

        name="Port2_Sheet"

    )



    hfss.create_lumped_port(

        input_object=port2_sheet.name,

        axisdir="-Z",

        portname="P2",

        renorm=True,

        renorm_impedance=50

    )



    hfss.save_project()

    print(f"Project saved: {hfss.project_file}")

   

    return hfss



if __name__ == "__main__":

    # Example usage similar to pixel_gds.py

    rows, cols = 5, 5

    matrix = generate_design_matrix(rows, cols)

   

    # Run HFSS

    # Note: Ensure Ansys Electronics Desktop is installed and licenced.

    try:

        hfss = draw_pixelated_device_hfss(

            design_matrix=matrix,

            non_graphical=False,  # Set to True if you don't want the UI to open

            design_name="Pixelated_Device_Demo"

        )

        print("Done.")

    except Exception as e:

        print(f"Failed to run HFSS integration: {e}")

        print("Ensure 'ansys-aedt-core' is installed and Ansys EDT is available.")