from pyntcloud import PyntCloud
import matplotlib.pyplot as plt

# Load the PLY file
ply_file = "/home/csslab/Documents/Aqsa/Mast3r/mast3r/checkpoints/mast3r_demo/tmp8ws94yep_scene.ply"  # Replace with the actual path to your PLY file
cloud = PyntCloud.from_file(ply_file)


# Optionally, you can display the first few rows of the point cloud
print(cloud.points.head())  # Display the first few points in the point cloud
# Show some information about the point cloud
print(cloud)  # This prints metadata about the point cloud



