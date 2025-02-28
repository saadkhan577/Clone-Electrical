import os

# Define paths using os.path.join for platform independence
image_path = os.path.join("C:", "Users", "haier", "Documents", "clone Electrical", "Floor-Plan-Detection-main", "Images", "example.png")
target_path = os.path.join("C:", "Users", "haier", "Documents", "clone Electrical", "Floor-Plan-Detection-main")  # will export in two formats (.blend and .stl)
program_path = os.getcwd()
blender_install_path = os.path.join(program_path, "2.93.1", "blender")
blender_script_path = os.path.join(program_path, "C:", "Users", "haier", "Documents", "clone Electrical", "Floor-Plan-Detection-main", "floorplan_to_3dObject_in_blender.py")

# Super-Resolution settings
SR_scale = 2
SR_method = 'lapsrn'

# CubiCasa setting
CubiCasa = True
