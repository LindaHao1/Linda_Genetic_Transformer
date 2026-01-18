import sys
import re

def layermap_to_python(input_file, output_file):
    print(f"Reading {input_file}...")
    
    # Store definitions as a dictionary: {var_name: code_line}
    # This lets us overwrite or check for duplicates easily.
    definitions = {}
    
    # Track used variable names to ensure uniqueness
    # Format: {var_name: (layer, datatype)}
    seen_vars = {}

    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments/empty lines
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 4:
                continue

            # Extract raw data
            raw_layer_name = parts[0]
            purpose = parts[1]
            try:
                layer_num = int(parts[2])
                datatype = int(parts[3])
            except ValueError:
                continue

            # --- 1. Base Name Construction ---
            if purpose == 'drawing':
                # Default "drawing" purpose usually gets the clean name
                base_name = raw_layer_name.upper()
            else:
                base_name = f"{raw_layer_name}_{purpose}".upper()

            # Sanitize characters (dots, dashes become underscores)
            clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)

            # Handle "Starts with Number" case
            note_comment = ""
            if clean_name[0].isdigit():
                clean_name = f"L_{clean_name}"
                note_comment = f"  # Fixed: Was '{raw_layer_name}'"

            # --- 2. Conflict Handling & Uniqueness ---
            final_var_name = clean_name
            
            # If this variable name is already taken...
            if final_var_name in seen_vars:
                existing_layer, existing_dtype = seen_vars[final_var_name]
                
                # CASE A: Exact duplicate? (Same name, same layer, same datatype)
                if (layer_num, datatype) == (existing_layer, existing_dtype):
                    # It's just a redundant line in the file. Skip it.
                    continue
                
                # CASE B: Priority overwrite? 
                # If the new one is Datatype 0 (the "Main" drawing layer), we want it to win.
                if datatype == 0 and existing_dtype != 0:
                    # Overwrite the previous entry with this better one
                    pass 
                
                # CASE C: The name is taken, so we must rename this new one.
                # We append the datatype to make it unique (e.g. M1_235)
                else:
                    final_var_name = f"{clean_name}_{datatype}"
                    note_comment += f"  # Renamed due to collision"

            # Store it
            seen_vars[final_var_name] = (layer_num, datatype)
            
            # Format the code line
            code_line = f"    {final_var_name} = ({layer_num}, {datatype}){note_comment}"
            definitions[final_var_name] = code_line

    # --- 3. Write Output ---
    with open(output_file, 'w') as f:
        f.write("import gdsfactory as gf\n\n")
        f.write("class LayerMap(gf.Technology.LayerMap):\n")
        f.write('    """\n    Auto-generated LayerMap for GF 22FDX.\n')
        f.write('    NOTE: Duplicate names were suffixed with datatypes.\n    """\n')
        
        # Sort by name so the file is readable
        for var_name in sorted(definitions.keys()):
            f.write(definitions[var_name] + "\n")
            
        f.write("\n")
        # f.write("LAYER = LayerMap()\n")

    print(f"Success! Generated {output_file} with {len(definitions)} unique layers.")

# --- EXECUTION ---
if __name__ == "__main__":
    layermap_to_python('cmos22fdsoi_tech.layermap', 'layers.py')