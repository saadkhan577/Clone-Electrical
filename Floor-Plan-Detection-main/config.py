import os

image_path = r"C:\Users\haier\Documents\clone Electrical\Floor-Plan-Detection-main\Images\example.png"
target_path = r"C:\Users\haier\Documents\clone Electrical\Floor-Plan-Detection-main" # will export in two formats (.blend and .stl)
program_path = os.getcwd()  
blender_install_path = program_path+"\2.93.1\blender"
blender_script_path = program_path+r"C:\Users\haier\Documents\clone Electrical\Floor-Plan-Detection-main\floorplan_to_3dObject_in_blender.py"

SR_scale = 2
SR_method = 'lapsrn'

CubiCasa = True