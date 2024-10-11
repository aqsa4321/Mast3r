#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# sparse gradio demo functions
# --------------------------------------------------------
import math
import gradio
import os
import numpy as np
import functools
import trimesh
import copy
from scipy.spatial.transform import Rotation
import tempfile
import shutil
from plyfile import PlyData, PlyElement
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.cloud_opt.tsdf_optimizer import TSDFPostProcess

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.demo import get_args_parser as dust3r_get_args_parser
import plyfile
import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as pl

pathgs = '/home/csslab/Documents/Aqsa/GS/dataexamples/family/mast3r/sparse/0'
# Check if the directory exists, if not create it
if not os.path.exists(pathgs):
    os.makedirs(pathgs)
    print(f"Directory {pathgs} created")
else:
    print(f"Directory {pathgs} already exists, skipping creation.")
class SparseGAState():
    def __init__(self, sparse_ga, should_delete=False, cache_dir=None, outfile_name=None):
        self.sparse_ga = sparse_ga
        self.cache_dir = cache_dir
        self.outfile_name = outfile_name
        self.should_delete = should_delete

    def __del__(self):
        if not self.should_delete:
            return
        if self.cache_dir is not None and os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        self.cache_dir = None
        if self.outfile_name is not None and os.path.isfile(self.outfile_name):
            os.remove(self.outfile_name)
        self.outfile_name = None


def get_args_parser():
    parser = dust3r_get_args_parser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--gradio_delete_cache', default=None, type=int,
                        help='age/frequency at which gradio removes the file. If >0, matching cache is purged')

    actions = parser._actions
    for action in actions:
        if action.dest == 'model_name':
            action.choices = ["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"]
    # change defaults
    parser.prog = 'mast3r demo'
    return parser

def save_scene_as_ply(save_dir, outfile, pts3d, cams2world, focals):

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    # Save camera extrinsics (cams2world) to a separate .txt file
    cam_extrinsics_file = os.path.join(save_dir, 'cam_extrinsics.txt')
    with open(cam_extrinsics_file, 'w') as cam_file:
        for i, pose in enumerate(cams2world):
            cam_file.write(f'Camera {i} extrinsics:\n{pose}\n')
    print(f"Camera extrinsics saved in {cam_extrinsics_file}")

    # Save camera intrinsics (focals) to a separate .txt file
    cam_intrinsics_file = os.path.join(save_dir, 'cam_intrinsics.txt')
    with open(cam_intrinsics_file, 'w') as cam_file:
        for i, focal in enumerate(focals):
            cam_file.write(f'Camera {i} focal length: {focal}\n')
    print(f"Camera intrinsics saved in {cam_intrinsics_file}")

# Function to calculate FOV (Field of View)
def calculate_fov(focal_length, dimension):
    return 2 * math.atan(dimension / (2 * focal_length))

# Function to parse camera intrinsic file
def parse_intrinsics(file_path):
    intrinsics = {}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('Camera'):
                cam_id = int(line.split()[1])
                focal_length = float(line.split(':')[1].strip())
                intrinsics[cam_id] = focal_length
    return intrinsics

# Function to parse camera extrinsic file
# Function to parse camera extrinsic file
def parse_extrinsics(file_path):
    extrinsics = {}
    cam_id = None
    matrix = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Camera'):  # Start of new camera data
                if cam_id is not None and len(matrix) == 3:  # If we've read a previous camera's data (3 rows)
                    extrinsics[cam_id] = np.array(matrix)  # Save the 3x4 matrix
                cam_id = int(line.split()[1])  # Now set the new cam_id
                matrix = []  # Reset matrix for the new camera
            elif 'tensor([' in line:  # Handle the first row embedded in the 'tensor([' line
                clean_line = line.split('tensor([')[1].strip('[]').replace(',', '').replace(')', '').replace(']', '').strip()
                row = [float(x) for x in clean_line.split()]
                matrix.append(row)  # Add the first row
            elif line.startswith('['):  # Handle subsequent rows
                clean_line = line.strip('[]').replace(',', '').replace(')', '').replace(']', '').strip()
                row = [float(x) for x in clean_line.split()]
                if len(row) == 4 and len(matrix) < 3:  # Only take first 3 rows with 4 elements
                    matrix.append(row)
            elif 'tensor' in line or line.startswith(']'):
                continue  # Skip lines with 'tensor' or closing brackets

        # Add the last camera's data
        if cam_id is not None and len(matrix) == 3:
            extrinsics[cam_id] = np.array(matrix)  # Save the 3x4 matrix

    return extrinsics


def parse_extrinsics1(file_path):
    extrinsics = {}
    cam_id = None
    matrix = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Camera'):  # Start of new camera data
                if cam_id is not None:  # If we've read a previous camera's data
                    extrinsics[cam_id] = np.array(matrix).reshape(3, 4)  # Save the previous camera's matrix
                cam_id = int(line.split()[1])  # Now set the new cam_id
                matrix = []  # Reset matrix for the new camera
            elif line.startswith('tensor'):  # Skip 'tensor' lines
                continue
            elif line:  # Matrix data line
                # Clean the line by removing unwanted characters
                clean_line = line.strip('[]').replace(',', '').replace(']', '').replace('(', '').replace(')', '').strip()
                try:
                    matrix.append([float(x) for x in clean_line.split()])
                except ValueError as e:
                    print(f"Error converting to float: {e}")
                    print(f"Offending line: {clean_line}")
                    continue

        # Add the last camera's data
        if cam_id is not None and matrix:
            extrinsics[cam_id] = np.array(matrix).reshape(3, 4)

    return extrinsics

# Main function to generate camera info file in the correct format
def generate_camera_info(images_path, cam_intrinsics_file, cam_extrinsics_file, output_path):
    # Parse intrinsic and extrinsic data
    intrinsics = parse_intrinsics(cam_intrinsics_file)
    extrinsics = parse_extrinsics(cam_extrinsics_file)

    # Get the list of images
    image_files = sorted([f for f in os.listdir(images_path) if f.endswith('.jpg') or f.endswith('.png')])

    # Ensure the number of images matches the intrinsic/extrinsic data
    if len(image_files) != len(intrinsics) or len(image_files) != len(extrinsics):
        raise ValueError("Mismatch between the number of images and camera data.")
    output_file = os.path.join(output_path, 'cam_infos_created.txt')
    with open(output_file, 'w') as f_out:
        for i, image_file in enumerate(image_files):
            # Load the image to get its dimensions
            image_path = os.path.join(images_path, image_file)
            with Image.open(image_path) as img:
                width, height = img.size

            # Get focal length for current camera
            focal_length = intrinsics[i]

            # Calculate FOVX and FOVY
            fov_x = calculate_fov(focal_length, width)
            fov_y = calculate_fov(focal_length, height)

            # Get extrinsic data for the current camera
            extrinsic_matrix = extrinsics[i]
            rotation_matrix = extrinsic_matrix[:, :3]  # Get the rotation matrix (no need to convert to list)
            translation_vector = extrinsic_matrix[:, 3].tolist()  # Convert translation vector to a list

            # Write the camera info to the output file in the desired format
            f_out.write(f"Camera {i} Info:\n")
            f_out.write(f"UID: {i}\n")
            f_out.write("Rotation Matrix (R):\n")
            for row in rotation_matrix:  # Write the rotation matrix row by row, no commas
                f_out.write(f"[{' '.join([f'{val:.8f}' for val in row])}]\n")
            f_out.write(f"Translation Vector (T): [{' '.join([f'{val:.8f}' for val in translation_vector])}]\n")
            f_out.write(f"FOVY: {fov_y:.10f}\n")
            f_out.write(f"FOVX: {fov_x:.10f}\n")
            f_out.write(f"Image Path: {image_path}\n")
            f_out.write(f"Image Name: {os.path.basename(image_path)}\n")
            f_out.write(f"Width: {width}\n")
            f_out.write(f"Height: {height}\n")
            f_out.write("\n" + "-"*50 + "\n\n")

def compute_normals(points):
    """
    Compute the normals of a point cloud using Open3D.

    :param points: 3D points (Nx3 array).
    :return: Normals (Nx3 array).
    """
    # Convert points to Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Estimate the normals using Open3D
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))

    # Retrieve the computed normals
    normals = np.asarray(pcd.normals)
    return normals

def save_xyz_rgb_normals_as_ply(save_path, points, colors):
    normals = np.zeros_like(points)
    vertex = np.array([tuple(list(p) + list(n) + list(c)) for p, n, c in zip(points, normals, colors)],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    # Create the PlyElement and PlyData
    ply_data = plyfile.PlyElement.describe(vertex, 'vertex')
    ply_data = PlyData([ply_data])
    # Write the data to a PLY file
    with open(save_path, 'wb') as f:
        ply_data.write(f)

    print(f"Saved PLY file with normals to {save_path}")
def save_mesh_vertices_as_ply(save_path, mesh):

    # Extract the vertices (Nx3 array of XYZ coordinates)
    vertices = mesh.vertices

    # Get or compute normals (Nx3 array of normals)
    normals = np.zeros_like(vertices)  # If no normals, set to zeros

    # Extract or generate RGB colors (Nx3 array of RGB values)
    if mesh.visual.vertex_colors is not None:
        colors = mesh.visual.vertex_colors[:, :3]  # Only take RGB, ignore alpha if present
    else:
        colors = np.ones_like(vertices) * 255  # Default to white if no color is provided

    # Create the structured array for PLY
    vertex_data = np.array([tuple(list(v) + list(n) + list(c)) for v, n, c in zip(vertices, normals, colors)],
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                  ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    # Create the PlyElement and PlyData
    ply_element = PlyElement.describe(vertex_data, 'vertex')
    ply_data = PlyData([ply_element])

    # Write to the PLY file
    with open(save_path, 'wb') as f:
        ply_data.write(f)

    print(f"Saved mesh vertices to {save_path}")
def save_pointcloud_as_ply(pct, save_path):
    """
    Save a trimesh.PointCloud object as a .ply file with XYZ and RGB columns.

    Args:
        pct (trimesh.PointCloud): The point cloud object with points and colors.
        save_path (str): The file path to save the .ply file.
    """
    # Extract XYZ points from the point cloud
    points = pct.vertices  # Nx3 array of XYZ coordinates
    normals = np.zeros_like(points)  # If no normals, set to zeros
    # Extract RGB colors from the point cloud (assuming colors are in the range [0, 1] or [0, 255])
    colors = pct.colors[:, :3]

    # Create the structured array for PLY (with dtype for XYZ and RGB)
    vertex_data = np.array([tuple(list(v) + list(n) + list(c)) for v, n, c in zip(points, normals, colors)],
                           dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                  ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    # Create the PlyElement and PlyData
    ply_element = PlyElement.describe(vertex_data, 'vertex')
    ply_data = PlyData([ply_element])

    # Write the data to a PLY file
    with open(save_path, 'wb') as f:
        ply_data.write(f)

    print(f"Saved point cloud to {save_path}")



def _convert_scene_output_to_glb(outfile, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()
    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, mask)]).reshape(-1, 3)
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)]).reshape(-1, 3)
        valid_msk = np.isfinite(pts.sum(axis=1))
        normals = compute_normals(pts[valid_msk])
        path_file_points = os.path.join(pathgs, 'points.ply')
        pct = trimesh.PointCloud(pts[valid_msk], colors=col[valid_msk])
        save_pointcloud_as_ply(pct, path_file_points)
        #save_xyz_rgb_normals_as_ply(path_file_points, pts[valid_msk], (col[valid_msk] * 255).astype(np.uint8))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            pts3d_i = pts3d[i].reshape(imgs[i].shape)
            msk_i = mask[i] & np.isfinite(pts3d_i.sum(axis=-1))
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d_i, msk_i))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        path_file_points1 = os.path.join(pathgs, 'points_mesh.ply')
        save_mesh_vertices_as_ply(path_file_points1, mesh)
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile
def get_shape(nested_list):
    """
    Recursively determines the shape of a nested list.
    :param nested_list: A list (or nested list)
    :return: A tuple representing the shape of the nested list
    """
    if isinstance(nested_list, list):
        if len(nested_list) == 0:
            return (0,)
        else:
            return (len(nested_list),) + get_shape(nested_list[0])
    else:
        return ()  # Base case: if it's not a list, return an empty tuple


def get_3D_model_from_scene(silent, scene_state, min_conf_thr=2, as_pointcloud=False, mask_sky=False,
                            clean_depth=False, transparent_cams=False, cam_size=0.05, TSDF_thresh=0.75):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene_state is None:
        return None
    outfile = scene_state.outfile_name
    if outfile is None:
        return None

    # get optimized values from scene
    scene = scene_state.sparse_ga
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    cams2world = scene.get_im_poses().cpu()

    # Print dimensions of rgbimg, focals, cams2world
    shape = get_shape(rgbimg)
    print("Shape of the nested list:", shape)
    print(f"Dimensions of rgbimg: {len(rgbimg)}")
    print(f"Dimensions of rgbimg: {len(rgbimg[0])}")
    print(f"Dimensions of rgbimg: {len(rgbimg[1])}")
    #print(rgbimg)
    print(f"Dimensions of focals: {focals.shape}")
    print(f"Dimensions of cams2world: {cams2world.shape}")
    # 3D pointcloud from depthmap, poses and intrinsics
    if TSDF_thresh > 0:
        print("Aqsa")
        tsdf = TSDFPostProcess(scene, TSDF_thresh=TSDF_thresh)
        pts3d, _, confs = to_numpy(tsdf.get_dense_pts3d(clean_depth=clean_depth))
    else:
        pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=clean_depth))

    # Print dimensions of pts3d
    shape = get_shape(pts3d)
    print("Shape of the nested list:", shape)
    print(f"Dimensions of pts3d: {len(pts3d)}")
    print(f"Dimensions of pts3d: {len(pts3d[0][0])}")
    print(f"Dimensions of pts3d: {len(pts3d[1])}")
    print(f"Shape of rgbimg: {np.array(rgbimg).shape}")
    print(f"Shape of pts3d: {np.array(pts3d).shape}")
    msk = to_numpy([c > min_conf_thr for c in confs])
    outfile = _convert_scene_output_to_glb(outfile, rgbimg, pts3d, msk, focals, cams2world,
                                           as_pointcloud=as_pointcloud, transparent_cams=transparent_cams,
                                           cam_size=cam_size, silent=silent)
    print("Outmodel (outfile path):", outfile)  # Print the path to the output model file
    save_scene_as_ply(pathgs, outfile, pts3d, cams2world, focals)
    subdir = os.path.dirname(os.path.dirname(pathgs))
    images_folder = os.path.join(subdir, 'images')
    cam_intrinsics_file = os.path.join(pathgs, 'cam_intrinsics.txt')
    cam_extrinsics_file = os.path.join(pathgs, 'cam_extrinsics.txt')
    generate_camera_info(images_folder, cam_intrinsics_file, cam_extrinsics_file, pathgs)

    return outfile


def get_reconstructed_scene(outdir, gradio_delete_cache, model, device, silent, image_size, current_scene_state,
                            filelist, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size, scenegraph_type, winsize,
                            win_cyclic, refid, TSDF_thresh, shared_intrinsics, **kw):
    """
    from a list of images, run mast3r inference, sparse global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    image_names = filelist
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
        filelist = [filelist[0], filelist[0] + '_2']

    scene_graph_params = [scenegraph_type]
    if scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(winsize))
    elif scenegraph_type == "oneref":
        scene_graph_params.append(str(refid))
    if scenegraph_type in ["swin", "logwin"] and not win_cyclic:
        scene_graph_params.append('noncyclic')
    scene_graph = '-'.join(scene_graph_params)
    pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
    if optim_level == 'coarse':
        niter2 = 0
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.cache_dir is not None:
        cache_dir = current_scene_state.cache_dir
    elif gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix='_cache', dir=outdir)
    else:
        cache_dir = os.path.join(outdir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(filelist, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=shared_intrinsics,
                                    matching_conf_thr=matching_conf_thr, **kw)
    if current_scene_state is not None and \
        not current_scene_state.should_delete and \
            current_scene_state.outfile_name is not None:
        outfile_name = current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix='_scene.glb', dir=outdir)

    scene_state = SparseGAState(scene, gradio_delete_cache, cache_dir, outfile_name)
    print(image_names)

    outfile = get_3D_model_from_scene(silent, scene_state, min_conf_thr, as_pointcloud, mask_sky,
                                      clean_depth, transparent_cams, cam_size, TSDF_thresh)
    return scene_state, outfile


def set_scenegraph_options(inputfiles, win_cyclic, refid, scenegraph_type):
    num_files = len(inputfiles) if inputfiles is not None else 1
    show_win_controls = scenegraph_type in ["swin", "logwin"]
    show_winsize = scenegraph_type in ["swin", "logwin"]
    show_cyclic = scenegraph_type in ["swin", "logwin"]
    max_winsize, min_winsize = 1, 1
    if scenegraph_type == "swin":
        if win_cyclic:
            max_winsize = max(1, math.ceil((num_files - 1) / 2))
        else:
            max_winsize = num_files - 1
    elif scenegraph_type == "logwin":
        if win_cyclic:
            half_size = math.ceil((num_files - 1) / 2)
            max_winsize = max(1, math.ceil(math.log(half_size, 2)))
        else:
            max_winsize = max(1, math.ceil(math.log(num_files, 2)))
    winsize = gradio.Slider(label="Scene Graph: Window Size", value=max_winsize,
                            minimum=min_winsize, maximum=max_winsize, step=1, visible=show_winsize)
    win_cyclic = gradio.Checkbox(value=win_cyclic, label="Cyclic sequence", visible=show_cyclic)
    win_col = gradio.Column(visible=show_win_controls)
    refid = gradio.Slider(label="Scene Graph: Id", value=0, minimum=0,
                          maximum=num_files - 1, step=1, visible=scenegraph_type == 'oneref')
    return win_col, winsize, win_cyclic, refid


def main_demo(tmpdirname, model, device, image_size, server_name, server_port, silent=False,
              share=False, gradio_delete_cache=False):
    if not silent:
        print('Outputing stuff in', tmpdirname)

    recon_fun = functools.partial(get_reconstructed_scene, tmpdirname, gradio_delete_cache, model, device,
                                  silent, image_size)
    model_from_scene_fun = functools.partial(get_3D_model_from_scene, silent)

    def get_context(delete_cache):
        css = """.gradio-container {margin: 0 !important; min-width: 100%};"""
        title = "MASt3R Demo"
        if delete_cache:
            return gradio.Blocks(css=css, title=title, delete_cache=(delete_cache, delete_cache))
        else:
            return gradio.Blocks(css=css, title="MASt3R Demo")  # for compatibility with older versions

    with get_context(gradio_delete_cache) as demo:
        # scene state is save so that you can change conf_thr, cam_size... without rerunning the inference
        scene = gradio.State(None)
        gradio.HTML('<h2 style="text-align: center;">MASt3R Demo</h2>')
        with gradio.Column():
            inputfiles = gradio.File(file_count="multiple")
            with gradio.Row():
                with gradio.Column():
                    with gradio.Row():
                        lr1 = gradio.Slider(label="Coarse LR", value=0.07, minimum=0.01, maximum=0.2, step=0.01)
                        niter1 = gradio.Number(value=500, precision=0, minimum=0, maximum=10_000,
                                               label="num_iterations", info="For coarse alignment!")
                        lr2 = gradio.Slider(label="Fine LR", value=0.014, minimum=0.005, maximum=0.05, step=0.001)
                        niter2 = gradio.Number(value=200, precision=0, minimum=0, maximum=100_000,
                                               label="num_iterations", info="For refinement!")
                        optim_level = gradio.Dropdown(["coarse", "refine", "refine+depth"],
                                                      value='refine', label="OptLevel",
                                                      info="Optimization level")
                    with gradio.Row():
                        matching_conf_thr = gradio.Slider(label="Matching Confidence Thr", value=5.,
                                                          minimum=0., maximum=30., step=0.1,
                                                          info="Before Fallback to Regr3D!")
                        shared_intrinsics = gradio.Checkbox(value=False, label="Shared intrinsics",
                                                            info="Only optimize one set of intrinsics for all views")
                        scenegraph_type = gradio.Dropdown([("complete: all possible image pairs", "complete"),
                                                           ("swin: sliding window", "swin"),
                                                           ("logwin: sliding window with long range", "logwin"),
                                                           ("oneref: match one image with all", "oneref")],
                                                          value='complete', label="Scenegraph",
                                                          info="Define how to make pairs",
                                                          interactive=True)
                        with gradio.Column(visible=False) as win_col:
                            winsize = gradio.Slider(label="Scene Graph: Window Size", value=1,
                                                    minimum=1, maximum=1, step=1)
                            win_cyclic = gradio.Checkbox(value=False, label="Cyclic sequence")
                        refid = gradio.Slider(label="Scene Graph: Id", value=0,
                                              minimum=0, maximum=0, step=1, visible=False)
            run_btn = gradio.Button("Run")

            with gradio.Row():
                # adjust the confidence threshold
                min_conf_thr = gradio.Slider(label="min_conf_thr", value=1.5, minimum=0.0, maximum=10, step=0.1)
                # adjust the camera size in the output pointcloud
                cam_size = gradio.Slider(label="cam_size", value=0.2, minimum=0.001, maximum=1.0, step=0.001)
                TSDF_thresh = gradio.Slider(label="TSDF Threshold", value=0., minimum=0., maximum=1., step=0.01)
            with gradio.Row():
                as_pointcloud = gradio.Checkbox(value=True, label="As pointcloud")
                # two post process implemented
                mask_sky = gradio.Checkbox(value=False, label="Mask sky")
                clean_depth = gradio.Checkbox(value=True, label="Clean-up depthmaps")
                transparent_cams = gradio.Checkbox(value=False, label="Transparent cameras")

            outmodel = gradio.Model3D()

            # events
            scenegraph_type.change(set_scenegraph_options,
                                   inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                                   outputs=[win_col, winsize, win_cyclic, refid])
            inputfiles.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[win_col, winsize, win_cyclic, refid])
            win_cyclic.change(set_scenegraph_options,
                              inputs=[inputfiles, win_cyclic, refid, scenegraph_type],
                              outputs=[win_col, winsize, win_cyclic, refid])
            run_btn.click(fn=recon_fun,
                          inputs=[scene, inputfiles, optim_level, lr1, niter1, lr2, niter2, min_conf_thr, matching_conf_thr,
                                  as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                                  scenegraph_type, winsize, win_cyclic, refid, TSDF_thresh, shared_intrinsics],
                          outputs=[scene, outmodel])
            min_conf_thr.release(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                 outputs=outmodel)
            cam_size.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, TSDF_thresh],
                            outputs=outmodel)
            TSDF_thresh.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, TSDF_thresh],
                               outputs=outmodel)
            as_pointcloud.change(fn=model_from_scene_fun,
                                 inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                         clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                 outputs=outmodel)
            mask_sky.change(fn=model_from_scene_fun,
                            inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                    clean_depth, transparent_cams, cam_size, TSDF_thresh],
                            outputs=outmodel)
            clean_depth.change(fn=model_from_scene_fun,
                               inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                       clean_depth, transparent_cams, cam_size, TSDF_thresh],
                               outputs=outmodel)
            transparent_cams.change(model_from_scene_fun,
                                    inputs=[scene, min_conf_thr, as_pointcloud, mask_sky,
                                            clean_depth, transparent_cams, cam_size, TSDF_thresh],
                                    outputs=outmodel)
    demo.launch(share=True, server_name=server_name, server_port=server_port)
