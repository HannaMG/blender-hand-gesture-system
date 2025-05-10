import bpy
import csv
import os

# Change file name for each test
path = "test_result.csv"

# Open csv file 
with open(path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    # Header row
    writer.writerow(["object_name", "vertex_index", "x", "y", "z"])
    
    # Iterate over all objects in the scene
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            
            # Iterate over each vertex
            for i, vertex in enumerate(obj.data.vertices):
                # Convert vertex coordinate from object to world space.
                world_coord = obj.matrix_world @ vertex.co
                writer.writerow([obj.name, i, world_coord.x, world_coord.y, world_coord.z])


print(f"Vertex data saved to {path}")