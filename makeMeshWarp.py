import time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
from scipy import ndimage
from skimage.util.shape import view_as_blocks
from pathlib import Path
import open3d as o3d
from tabulate import tabulate
import h5py
import cupy as cp  # Added for GPU-accelerated array operations

# Import and initialize NVIDIA Warp for GPU acceleration
import warp as wp

wp.init()
DEVICE = "cuda"

# Warp kernel to compute voxel corners and connectivity
@wp.kernel
def process_voxel_kernel(n: int,
                         true_indices: wp.array(dtype=wp.int32),
                         origin: wp.array(dtype=wp.float32),
                         v: float,
                         point_id_offset: int,
                         corners: wp.array(dtype=wp.float32),
                         connectivity: wp.array(dtype=wp.int32)):
    i = wp.tid()
    if i >= n:
        return

    base_idx = i * 3
    ix = int(true_indices[base_idx + 0])
    iy = int(true_indices[base_idx + 1])
    iz = int(true_indices[base_idx + 2])

    min0 = origin[0] + float(ix) * v
    min1 = origin[1] + float(iy) * v
    min2 = origin[2] + float(iz) * v
    max0 = min0 + v
    max1 = min1 + v
    max2 = min2 + v

    offset = i * 24
    corners[offset + 0] = min0
    corners[offset + 1] = min1
    corners[offset + 2] = min2
    corners[offset + 3] = max0
    corners[offset + 4] = min1
    corners[offset + 5] = min2
    corners[offset + 6] = max0
    corners[offset + 7] = max1
    corners[offset + 8] = min2
    corners[offset + 9]  = min0
    corners[offset + 10] = max1
    corners[offset + 11] = min2
    corners[offset + 12] = min0
    corners[offset + 13] = min1
    corners[offset + 14] = max2
    corners[offset + 15] = max0
    corners[offset + 16] = min1
    corners[offset + 17] = max2
    corners[offset + 18] = max0
    corners[offset + 19] = max1
    corners[offset + 20] = max2
    corners[offset + 21] = min0
    corners[offset + 22] = max1
    corners[offset + 23] = max2

    for j in range(8):
        connectivity[i * 8 + j] = point_id_offset + i * 8 + j

# Process a mesh level on the GPU, returning Warp arrays
def process_level_gpu(args):
    data, v, origin, level, point_id_offset = args
    true_indices = np.argwhere(data)  # Still computed on CPU due to np.argwhere
    num_voxels = len(true_indices)
    if num_voxels == 0:
        # Return empty Warp arrays instead of NumPy
        corners_wp = wp.zeros((0, 3), dtype=wp.vec3f, device=DEVICE)
        connectivity_wp = wp.zeros((0, 8), dtype=wp.int32, device=DEVICE)
        return corners_wp, connectivity_wp, level

    corners_flat = wp.zeros(num_voxels * 8 * 3, dtype=wp.float32, device=DEVICE)
    connectivity_flat = wp.zeros(num_voxels * 8, dtype=wp.int32, device=DEVICE)

    true_indices_flat = np.ascontiguousarray(true_indices.astype(np.int32)).flatten()
    wp_true_indices = wp.array(true_indices_flat, dtype=wp.int32, device=DEVICE)
    origin_arr = wp.array(np.array(origin, dtype=np.float32), dtype=wp.float32, device=DEVICE)

    wp.launch(
        kernel=process_voxel_kernel,
        dim=num_voxels,
        inputs=[num_voxels, wp_true_indices, origin_arr, v, point_id_offset],
        outputs=[corners_flat, connectivity_flat],
        device=DEVICE,
    )
    wp.synchronize()

    # Reshape on GPU without converting to NumPy
    corners_wp = corners_flat.reshape((num_voxels * 8, 3))
    connectivity_wp = connectivity_flat.reshape((num_voxels, 8))
    return corners_wp, connectivity_wp, level

# Warp kernels for voxel operations: grow, fill, remove, and crop

# Copy input matrix into a padded matrix on GPU
@wp.kernel
def copy_to_padded_kernel(
    input: wp.array3d(dtype=wp.uint8),
    padded: wp.array3d(dtype=wp.uint8),
    pad_x: int,
    pad_y: int,
    pad_z: int
):
    i, j, k = wp.tid()
    if i < input.shape[0] and j < input.shape[1] and k < input.shape[2]:
        padded[i + pad_x, j + pad_y, k + pad_z] = input[i, j, k]

# Apply convolution to a padded matrix on GPU
@wp.kernel
def convolution_kernel(
    padded: wp.array3d(dtype=wp.uint8),
    kernel: wp.array3d(dtype=wp.uint8),
    output: wp.array3d(dtype=wp.uint8),
    kx: int,
    ky: int,
    kz: int
):
    i, j, k = wp.tid()
    if i >= padded.shape[0] or j >= padded.shape[1] or k >= padded.shape[2]:
        return
    sum_val = wp.uint8(0)
    for di in range(kx):
        for dj in range(ky):
            for dk in range(kz):
                if kernel[di, dj, dk] != 0:
                    ii = i + di - kx // 2
                    jj = j + dj - ky // 2
                    kk = k + dk - kz // 2
                    if 0 <= ii < padded.shape[0] and 0 <= jj < padded.shape[1] and 0 <= kk < padded.shape[2]:
                        sum_val += padded[ii, jj, kk]
    if sum_val > 0:
        output[i, j, k] = wp.uint8(1)
    else:
        output[i, j, k] = wp.uint8(0)

# Expand a voxel matrix using convolution-based growth on GPU
def grow_gpu(matrix, voxSize, origin, kernel):
    pad = (np.array(kernel.shape) * 0.5).astype(int)
    print("    Grow padding: ", pad)
    wp_matrix = wp.array(matrix.astype(np.uint8), dtype=wp.uint8, device=DEVICE)
    wp_kernel = wp.array(kernel.astype(np.uint8), dtype=wp.uint8, device=DEVICE)
    padded_shape = tuple(np.array(matrix.shape) + 2 * pad)
    padded = wp.zeros(padded_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = matrix.shape
    wp.launch(
        kernel=copy_to_padded_kernel,
        dim=(nx, ny, nz),
        inputs=[wp_matrix, padded, pad[0], pad[1], pad[2]],
        device=DEVICE
    )
    output = wp.zeros(padded_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = padded_shape
    kx, ky, kz = kernel.shape
    wp.launch(
        kernel=convolution_kernel,
        dim=(nx, ny, nz),
        inputs=[padded, wp_kernel, output, kx, ky, kz],
        device=DEVICE
    )
    wp.synchronize()
    r = output.numpy().astype(bool)
    kernel_shape = np.array(kernel.shape)
    originPad = origin - (kernel_shape - 1) * voxSize * 0.5
    return r, originPad

# Compute OR of 2x2x2 blocks in a padded matrix on GPU
@wp.kernel
def compute_or_blocks(
    padded: wp.array3d(dtype=wp.uint8),
    or_results: wp.array3d(dtype=wp.uint8)
):
    bx, by, bz = wp.tid()
    if bx >= or_results.shape[0] or by >= or_results.shape[1] or bz >= or_results.shape[2]:
        return
    result = wp.uint8(0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                if padded[bx * 2 + di, by * 2 + dj, bz * 2 + dk] != 0:
                    result = wp.uint8(1)
                    break
            if result != 0:
                break
        if result != 0:
            break
    or_results[bx, by, bz] = result

# Perform binary dilation with a cross-shaped element on GPU
@wp.kernel
def binary_dilation_cross(
    input: wp.array3d(dtype=wp.uint8),
    output: wp.array3d(dtype=wp.uint8)
):
    i, j, k = wp.tid()
    if i >= input.shape[0] or j >= input.shape[1] or k >= input.shape[2]:
        return
    result = wp.uint8(0)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if (dx == 0 and dy == 0 and dz == 0) or (abs(dx) + abs(dy) + abs(dz) == 1):
                    ii = wp.clamp(i + dx, 0, input.shape[0] - 1)
                    jj = wp.clamp(j + dy, 0, input.shape[1] - 1)
                    kk = wp.clamp(k + dz, 0, input.shape[2] - 1)
                    if input[ii, jj, kk] != 0:
                        result = wp.uint8(1)
                        break
            if result != 0:
                break
        if result != 0:
            break
    output[i, j, k] = result

# Perform binary erosion with a cross-shaped element on GPU
@wp.kernel
def binary_erosion_cross(
    input: wp.array3d(dtype=wp.uint8),
    output: wp.array3d(dtype=wp.uint8)
):
    i, j, k = wp.tid()
    if i >= input.shape[0] or j >= input.shape[1] or k >= input.shape[2]:
        return
    result = wp.uint8(1)
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if (dx == 0 and dy == 0 and dz == 0) or (abs(dx) + abs(dy) + abs(dz) == 1):
                    ii = wp.clamp(i + dx, 0, input.shape[0] - 1)
                    jj = wp.clamp(j + dy, 0, input.shape[1] - 1)
                    kk = wp.clamp(k + dz, 0, input.shape[2] - 1)
                    if input[ii, jj, kk] == 0:
                        result = wp.uint8(0)
                        break
            if result == 0:
                break
        if result == 0:
            break
    output[i, j, k] = result

# Expand OR results back to the padded matrix size on GPU
@wp.kernel
def expand_or_results(
    or_results: wp.array3d(dtype=wp.uint8),
    padded: wp.array3d(dtype=wp.uint8)
):
    i, j, k = wp.tid()
    if i >= padded.shape[0] or j >= padded.shape[1] or k >= padded.shape[2]:
        return
    bx = i // 2
    by = j // 2
    bz = k // 2
    padded[i, j, k] = or_results[bx, by, bz]

# Extract the central region of a padded matrix on GPU
@wp.kernel
def extract_center(
    input: wp.array3d(dtype=wp.uint8),
    output: wp.array3d(dtype=wp.uint8),
    px: int,
    py: int,
    pz: int
):
    i, j, k = wp.tid()
    if i < output.shape[0] and j < output.shape[1] and k < output.shape[2]:
        output[i, j, k] = input[i + px, j + py, k + pz]

# Pad an array with zeros on GPU
def pad_array_zeros_gpu(input_arr, pad_width):
    px, py, pz = pad_width
    shape = input_arr.shape
    padded_shape = (shape[0] + 2 * px, shape[1] + 2 * py, shape[2] + 2 * pz)
    padded = wp.zeros(padded_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = shape
    wp.launch(
        kernel=copy_to_padded_kernel,
        dim=(nx, ny, nz),
        inputs=[input_arr, padded, px, py, pz],
        device=DEVICE
    )
    return padded

# Fill a voxel matrix with optional closing operation on GPU
def fill_gpu(matrix, voxSize, origin, close):
    a = (origin / voxSize) % 2
    inds = np.isclose(a, np.round(a), atol=1e-8)
    a[inds] = np.round(a[inds])
    paddingLo = np.floor(a).astype(int)
    paddingHi = np.round((matrix.shape + paddingLo) % 2).astype(int)
    originPad = origin - paddingLo * voxSize
    wp_matrix = wp.array(matrix.astype(np.uint8), dtype=wp.uint8, device=DEVICE)
    padded_shape = tuple(np.array(matrix.shape) + paddingLo + paddingHi)
    padded = wp.zeros(padded_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = matrix.shape
    wp.launch(
        kernel=copy_to_padded_kernel,
        dim=(nx, ny, nz),
        inputs=[wp_matrix, padded, paddingLo[0], paddingLo[1], paddingLo[2]],
        device=DEVICE
    )
    or_results_shape = tuple(np.array(padded_shape) // 2)
    or_results = wp.zeros(or_results_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = or_results_shape
    wp.launch(
        kernel=compute_or_blocks,
        dim=(nx, ny, nz),
        inputs=[padded, or_results],
        device=DEVICE
    )
    if close:
        padded_or_results = pad_array_zeros_gpu(or_results, (2, 2, 2))
        nx, ny, nz = padded_or_results.shape
        dilated = wp.zeros(padded_or_results.shape, dtype=wp.uint8, device=DEVICE)
        wp.launch(
            kernel=binary_dilation_cross,
            dim=(nx, ny, nz),
            inputs=[padded_or_results, dilated],
            device=DEVICE
        )
        eroded = wp.zeros(padded_or_results.shape, dtype=wp.uint8, device=DEVICE)
        wp.launch(
            kernel=binary_erosion_cross,
            dim=(nx, ny, nz),
            inputs=[dilated, eroded],
            device=DEVICE
        )
        nx, ny, nz = or_results_shape
        new_or_results = wp.zeros(or_results_shape, dtype=wp.uint8, device=DEVICE)
        wp.launch(
            kernel=extract_center,
            dim=(nx, ny, nz),
            inputs=[eroded, new_or_results, 2, 2, 2],
            device=DEVICE
        )
        or_results = new_or_results
    nx, ny, nz = padded_shape
    wp.launch(
        kernel=expand_or_results,
        dim=(nx, ny, nz),
        inputs=[or_results, padded],
        device=DEVICE
    )
    wp.synchronize()
    m = padded.numpy().astype(bool)
    return m, originPad

# Set specified voxel indices to False on GPU
@wp.kernel
def set_false_kernel(
    matrix: wp.array3d(dtype=wp.uint8),
    indices: wp.array(dtype=wp.int32, ndim=2),
    offset: wp.array(dtype=wp.int32, ndim=1)
):
    tid = wp.tid()
    if tid >= indices.shape[0]:
        return
    ix = indices[tid, 0] + offset[0]
    iy = indices[tid, 1] + offset[1]
    iz = indices[tid, 2] + offset[2]
    if 0 <= ix < matrix.shape[0] and 0 <= iy < matrix.shape[1] and 0 <= iz < matrix.shape[2]:
        matrix[ix, iy, iz] = wp.uint8(0)

# Remove specified voxels from a matrix on GPU
def remove_gpu(matrix, origin, removeMat, removeOrigin, voxSize):
    offset = np.round((removeOrigin - origin) / voxSize).astype(int)
    removeIndices = np.argwhere(removeMat)
    if len(removeIndices) == 0:
        return np.copy(matrix)
    wp_matrix = wp.array(matrix.astype(np.uint8), dtype=wp.uint8, device=DEVICE)
    wp_indices = wp.array(removeIndices, dtype=wp.int32, device=DEVICE)
    wp_offset = wp.array(offset, dtype=wp.int32, device=DEVICE)
    wp.launch(
        kernel=set_false_kernel,
        dim=len(removeIndices),
        inputs=[wp_matrix, wp_indices, wp_offset],
        device=DEVICE
    )
    wp.synchronize()
    mat = wp_matrix.numpy().astype(bool)
    return mat

# Copy a cropped region from a matrix on GPU
@wp.kernel
def copy_cropped(
    mat: wp.array3d(dtype=wp.uint8),
    cropped: wp.array3d(dtype=wp.uint8),
    offset_x: int,
    offset_y: int,
    offset_z: int
):
    i, j, k = wp.tid()
    if i < cropped.shape[0] and j < cropped.shape[1] and k < cropped.shape[2]:
        cropped[i, j, k] = mat[i + offset_x, j + offset_y, k + offset_z]

# Crop a voxel matrix to a specified domain on GPU
def crop_gpu(mat, origin, domainMin, domainMax, v):
    cMin = np.round((domainMin - origin) / v).astype(int)
    cMax = np.round((domainMax - origin) / v).astype(int)
    cropMin = np.maximum(cMin, 0)
    cropMax = np.minimum(cMax, mat.shape)
    cropped_shape = tuple(cropMax - cropMin)
    if any(s <= 0 for s in cropped_shape):
        return np.empty((0, 0, 0), dtype=bool), origin
    wp_mat = wp.array(mat.astype(np.uint8), dtype=wp.uint8, device=DEVICE)
    wp_cropped = wp.zeros(cropped_shape, dtype=wp.uint8, device=DEVICE)
    nx, ny, nz = cropped_shape
    wp.launch(
        kernel=copy_cropped,
        dim=(nx, ny, nz),
        inputs=[wp_mat, wp_cropped, cropMin[0], cropMin[1], cropMin[2]],
        device=DEVICE
    )
    wp.synchronize()
    cropMat = wp_cropped.numpy().astype(bool)
    origin = origin + cropMin * v
    return cropMat, origin

# Warp kernels for merging duplicate points on GPU

# Compute grid coordinates based on point tolerance
@wp.kernel
def compute_grid_coords(
    coords: wp.array(dtype=wp.vec3f),
    tolerance: wp.float32,
    grid_coords: wp.array(dtype=wp.vec3i)
):
    i = wp.tid()
    if i < coords.shape[0]:
        c = coords[i]
        grid_coords[i] = wp.vec3i(
            wp.int32(wp.round(c[0] / tolerance)),
            wp.int32(wp.round(c[1] / tolerance)),
            wp.int32(wp.round(c[2] / tolerance))
        )

# Normalize grid coordinates by subtracting minimum values
@wp.kernel
def normalize_grid_coords(
    grid_coords: wp.array(dtype=wp.vec3i),
    min_coords: wp.array(dtype=wp.int32)
):
    i = wp.tid()
    if i < grid_coords.shape[0]:
        gc = grid_coords[i]
        grid_coords[i] = wp.vec3i(
            gc[0] - min_coords[0],
            gc[1] - min_coords[1],
            gc[2] - min_coords[2]
        )

# Convert 3D grid coordinates to linear indices
@wp.kernel
def compute_indices(
    grid_coords: wp.array(dtype=wp.vec3i),
    span: wp.array(dtype=wp.int64),
    indices: wp.array(dtype=wp.int64)
):
    i = wp.tid()
    if i < grid_coords.shape[0]:
        gc = grid_coords[i]
        x = wp.int64(gc[0])
        y = wp.int64(gc[1])
        z = wp.int64(gc[2])
        indices[i] = x + span[0] * (y + span[1] * z)

# Define a constant cross-shaped kernel for morphological operations
crosskernel = np.ones((3, 3, 3), bool)
crosskernel[0, 0, 0] = False
crosskernel[0, 2, 0] = False
crosskernel[0, 0, 2] = False
crosskernel[2, 0, 0] = False
crosskernel[2, 2, 0] = False
crosskernel[2, 0, 2] = False
crosskernel[2, 2, 2] = False

# Round values close to integers and floor them
def roundCloseFloor(a, intTol):
    inds = np.argwhere(np.isclose(a, np.round(a), atol=intTol))
    a[inds] = np.round(a[inds])
    return np.floor(a)

# Save mesh data to a VTK unstructured grid file with optimized merging
def save_to_unstructured_vtk(levels_data, filename, voxSize):
    print("/// Creating combined Unstructured Grid VTK file...")
    tic = time.perf_counter()
    points = vtk.vtkPoints()
    grid = vtk.vtkUnstructuredGrid()
    num_voxels_per_level = [np.sum(data) for data, _, _, _ in levels_data]
    num_points_per_level = [8 * num_voxels for num_voxels in num_voxels_per_level]
    point_id_offsets = np.cumsum([0] + num_points_per_level[:-1])
    all_corners_wp = []
    all_connectivity_wp = []
    all_levels = []
    total_cells = 0
    for level_idx, ((data, v, origin, level), offset) in enumerate(zip([item[:4] for item in levels_data], point_id_offsets)):
        corners_wp, connectivity_wp, lev = process_level_gpu((data, v, origin, level, offset))
        if num_voxels_per_level[level_idx] > 0:
            print(f"    Processing level {lev}: Voxel size {v}, Shape {data.shape}")
        else:
            print(f"    Skipping level {lev} (no unique data)")
        all_corners_wp.append(corners_wp)
        all_connectivity_wp.append(connectivity_wp)
        all_levels.extend([lev] * connectivity_wp.shape[0])
        total_cells += connectivity_wp.shape[0]
    print("    Processing levels complete")

    # Stack coordinates and connectivity on GPU using CuPy
    print("    Stacking coordinates and connectivity")
    all_coords_cp = cp.concatenate([cp.asarray(c) for c in all_corners_wp], axis=0)
    all_connectivity_cp = cp.concatenate([cp.asarray(c) for c in all_connectivity_wp], axis=0)

    # Optimized merging of duplicate points on GPU
    print("    Merging duplicate points")
    tic_counter = time.perf_counter()

    # Create Warp arrays from CuPy arrays using pointers
    ptr_coords = all_coords_cp.data.ptr  # Get the raw integer pointer
    all_corners_wp = wp.array(ptr=ptr_coords, shape=(all_coords_cp.shape[0],), dtype=wp.vec3f, device='cuda', owner=False)
    ptr_connectivity = all_connectivity_cp.data.ptr  # Get the raw integer pointer
    all_connectivity_wp = wp.array(ptr=ptr_connectivity, shape=all_connectivity_cp.shape, dtype=wp.int32, device='cuda', owner=False)

    tolerance = voxSize / 1000
    total_points = all_corners_wp.shape[0]

    # Compute grid coordinates
    grid_coords = wp.zeros(total_points, dtype=wp.vec3i, device="cuda")
    wp.launch(kernel=compute_grid_coords, dim=total_points, inputs=[all_corners_wp, tolerance], outputs=[grid_coords])

    # Extract components using CuPy
    grid_coords_cp = cp.asarray(grid_coords)
    x_coords_cp = grid_coords_cp[:, 0]
    y_coords_cp = grid_coords_cp[:, 1]
    z_coords_cp = grid_coords_cp[:, 2]

    # Compute min and max on GPU
    min_x = cp.min(x_coords_cp).item()
    min_y = cp.min(y_coords_cp).item()
    min_z = cp.min(z_coords_cp).item()
    max_x = cp.max(x_coords_cp).item()
    max_y = cp.max(y_coords_cp).item()
    max_z = cp.max(z_coords_cp).item()

    # Normalize grid coordinates
    min_coords = wp.array([min_x, min_y, min_z], dtype=wp.int32, device="cuda")
    wp.launch(kernel=normalize_grid_coords, dim=total_points, inputs=[grid_coords, min_coords])

    # Compute spans
    span = [max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1]
    span = [s if s > 0 else 1 for s in span]

    # Compute linear indices
    indices = wp.zeros(total_points, dtype=wp.int64, device="cuda")
    wp.launch(kernel=compute_indices, dim=total_points, inputs=[grid_coords, wp.array(span, dtype=wp.int64, device="cuda")], outputs=[indices])

    # Sort indices on GPU with CuPy
    indices_cp = cp.asarray(indices)
    sorted_order_cp = cp.argsort(indices_cp)
    sorted_indices_cp = indices_cp[sorted_order_cp]

    # Compute mask on GPU
    mask_cp = cp.zeros_like(sorted_indices_cp, dtype=cp.uint8)
    mask_cp[1:] = (sorted_indices_cp[1:] != sorted_indices_cp[:-1]).astype(cp.uint8)
    mask_cp[0] = 1

    # Compute prefix sum on GPU
    prefix_sum_cp = cp.cumsum(mask_cp, dtype=cp.int32)
    num_unique = prefix_sum_cp[-1].item()

    # Get first occurrence indices
    unique_indices_cp = cp.where(mask_cp)[0]
    first_occurrence_indices_cp = sorted_order_cp[unique_indices_cp]

    # Extract unique points
    unique_points_cp = all_coords_cp[first_occurrence_indices_cp]

    # Compute mapping with searchsorted
    unique_grid_indices_cp = sorted_indices_cp[unique_indices_cp]
    mapping_cp = cp.searchsorted(unique_grid_indices_cp, indices_cp, side='left')

    # Remap connectivity
    new_connectivity_cp = mapping_cp[all_connectivity_cp]

    # Transfer final results to host
    unique_points = unique_points_cp.get().astype(np.float32)
    new_connectivity = new_connectivity_cp.get()

    toc_counter = time.perf_counter()
    print(f"    Merged duplicate points in {toc_counter - tic_counter:0.1f} seconds")

    # Create connectivity array for VTK
    print("    Creating connectivity array")
    num_points_per_cell = np.full((total_cells, 1), 8, dtype=np.int64)
    full_connectivity = np.hstack((num_points_per_cell, new_connectivity)).flatten()
    full_connectivity_vtk = numpy_to_vtkIdTypeArray(full_connectivity, deep=True)
    cell_types = np.full(total_cells, vtk.VTK_HEXAHEDRON, dtype=np.uint8)
    cell_types_vtk = numpy_to_vtk(cell_types, deep=True)
    cells = vtk.vtkCellArray()
    cells.SetCells(total_cells, full_connectivity_vtk)
    vtk_coords = numpy_to_vtk(unique_points, deep=True)
    points.SetData(vtk_coords)
    grid.SetPoints(points)
    grid.SetCells(cell_types_vtk, cells)
    level_data_array_vtk = numpy_to_vtk(np.array(all_levels, dtype=np.uint8), deep=True)
    level_data_array_vtk.SetName("Level")
    grid.GetCellData().AddArray(level_data_array_vtk)

    # Write the VTK file
    print("    Writing Unstructured Grid VTU")
    tic_counter = time.perf_counter()
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetDataModeToBinary()
    writer.SetInputData(grid)
    writer.Write()
    toc_counter = time.perf_counter()
    toc = time.perf_counter()
    print(f"    Combined Unstructured Grid VTU file written in {toc_counter - tic_counter:0.1f} seconds")
    print(f"    Total output processing complete {toc - tic:0.1f} seconds")

# Save mesh data to an HDF5 file for XDMF visualization with optimized merging
def save_to_hdf5(levels_data, filename, voxSize):
    print("/// Creating combined HDF5 file for XDMF...")
    tic = time.perf_counter()
    num_voxels_per_level = [np.sum(data) for data, _, _, _ in levels_data]
    num_points_per_level = [8 * num_voxels for num_voxels in num_voxels_per_level]
    point_id_offsets = np.cumsum([0] + num_points_per_level[:-1])
    all_corners_wp = []
    all_connectivity_wp = []
    all_levels = []
    total_cells = 0
    for level_idx, (data, v, origin, level) in enumerate(levels_data):
        corners_wp, connectivity_wp, _ = process_level_gpu((data, v, origin, level, point_id_offsets[level_idx]))
        if num_voxels_per_level[level_idx] > 0:
            print(f"    Processing level {level}: Voxel size {v}, Origin {origin}, Shape {data.shape}")
        else:
            print(f"    Skipping level {level} (no unique data)")
        all_corners_wp.append(corners_wp)
        all_connectivity_wp.append(connectivity_wp)
        all_levels.extend([level] * connectivity_wp.shape[0])
        total_cells += connectivity_wp.shape[0]
    print("    Processing levels complete")

    # Stack coordinates and connectivity on GPU
    all_coords_cp = cp.concatenate([cp.asarray(c) for c in all_corners_wp], axis=0)
    all_connectivity_cp = cp.concatenate([cp.asarray(c) for c in all_connectivity_wp], axis=0)

    # Optimized merging of duplicate points on GPU
    print("    Merging duplicate points")
    tic_counter = time.perf_counter()

    # Create Warp arrays from CuPy arrays using pointers
    ptr_coords = all_coords_cp.data.ptr  # Get the raw integer pointer
    all_corners_wp = wp.array(ptr=ptr_coords, shape=(all_coords_cp.shape[0],), dtype=wp.vec3f, device='cuda', owner=False)
    ptr_connectivity = all_connectivity_cp.data.ptr  # Get the raw integer pointer
    num_elements_connectivity = all_connectivity_cp.size
    all_connectivity_wp = wp.array(ptr=ptr_connectivity, shape=all_connectivity_cp.shape, dtype=wp.int32, device='cuda', owner=False)

    tolerance = voxSize / 1000
    total_points = all_corners_wp.shape[0]

    # Compute grid coordinates
    grid_coords = wp.zeros(total_points, dtype=wp.vec3i, device="cuda")
    wp.launch(kernel=compute_grid_coords, dim=total_points, inputs=[all_corners_wp, tolerance], outputs=[grid_coords])

    # Extract components using CuPy
    grid_coords_cp = cp.asarray(grid_coords)
    x_coords_cp = grid_coords_cp[:, 0]
    y_coords_cp = grid_coords_cp[:, 1]
    z_coords_cp = grid_coords_cp[:, 2]

    # Compute min and max on GPU
    min_x = cp.min(x_coords_cp).item()
    min_y = cp.min(y_coords_cp).item()
    min_z = cp.min(z_coords_cp).item()
    max_x = cp.max(x_coords_cp).item()
    max_y = cp.max(y_coords_cp).item()
    max_z = cp.max(z_coords_cp).item()

    # Normalize grid coordinates
    min_coords = wp.array([min_x, min_y, min_z], dtype=wp.int32, device="cuda")
    wp.launch(kernel=normalize_grid_coords, dim=total_points, inputs=[grid_coords, min_coords])

    # Compute spans
    span = [max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1]
    span = [s if s > 0 else 1 for s in span]

    # Compute linear indices
    indices = wp.zeros(total_points, dtype=wp.int64, device="cuda")
    wp.launch(kernel=compute_indices, dim=total_points, inputs=[grid_coords, wp.array(span, dtype=wp.int64, device="cuda")], outputs=[indices])

    # Sort indices on GPU with CuPy
    indices_cp = cp.asarray(indices)
    sorted_order_cp = cp.argsort(indices_cp)
    sorted_indices_cp = indices_cp[sorted_order_cp]

    # Compute mask on GPU
    mask_cp = cp.zeros_like(sorted_indices_cp, dtype=cp.uint8)
    mask_cp[1:] = (sorted_indices_cp[1:] != sorted_indices_cp[:-1]).astype(cp.uint8)
    mask_cp[0] = 1

    # Compute prefix sum on GPU
    prefix_sum_cp = cp.cumsum(mask_cp, dtype=cp.int32)
    num_unique = prefix_sum_cp[-1].item()

    # Get first occurrence indices
    unique_indices_cp = cp.where(mask_cp)[0]
    first_occurrence_indices_cp = sorted_order_cp[unique_indices_cp]

    # Extract unique points
    unique_points_cp = all_coords_cp[first_occurrence_indices_cp]

    # Compute mapping with searchsorted
    unique_grid_indices_cp = sorted_indices_cp[unique_indices_cp]
    mapping_cp = cp.searchsorted(unique_grid_indices_cp, indices_cp, side='left')

    # Remap connectivity
    new_connectivity_cp = mapping_cp[all_connectivity_cp]

    # Transfer final results to host
    unique_points = unique_points_cp.get().astype(np.float32)
    connectivity = new_connectivity_cp.get()

    toc_counter = time.perf_counter()
    print(f"    Merged duplicate points in {toc_counter - tic_counter:0.1f} seconds")

    # Write HDF5 file
    print("    Writing HDF5 file")
    tic_counter = time.perf_counter()
    with h5py.File(filename, 'w') as f:
        f.create_dataset('/Mesh/Points', data=unique_points, compression='gzip', compression_opts=2)
        f.create_dataset('/Mesh/Connectivity', data=connectivity, compression='gzip', compression_opts=2)
        f.create_dataset('/Mesh/Level', data=np.array(all_levels, dtype=np.uint8), compression='gzip', compression_opts=2)
        f.attrs['voxel_size'] = voxSize
        f.attrs['total_cells'] = total_cells
        f.attrs['point_count'] = len(unique_points)
    toc_counter = time.perf_counter()
    toc = time.perf_counter()
    print(f"    HDF5 file written in {toc_counter - tic_counter:0.1f} seconds")
    print(f"    Total output processing complete in {toc - tic:0.1f} seconds")
    return total_cells, len(unique_points)

# Generate an XDMF file to accompany the HDF5 file
def save_xdmf(hdf5_filename, xdmf_filename, total_cells, num_points):
    print(f"/// Generating XDMF file: {xdmf_filename}")
    hdf5_rel_path = hdf5_filename.split('/')[-1]
    xdmf_content = f"""<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
    <Domain>
        <Grid Name="VoxelMesh" GridType="Uniform">
            <Topology TopologyType="Hexahedron" NumberOfElements="{total_cells}">
                <DataItem Dimensions="{total_cells} 8" NumberType="Int" Format="HDF">
                    {hdf5_rel_path}:/Mesh/Connectivity
                </DataItem>
            </Topology>
            <Geometry GeometryType="XYZ">
                <DataItem Dimensions="{num_points} 3" NumberType="Float" Precision="4" Format="HDF">
                    {hdf5_rel_path}:/Mesh/Points
                </DataItem>
            </Geometry>
            <Attribute Name="Level" AttributeType="Scalar" Center="Cell">
                <DataItem Dimensions="{total_cells}" NumberType="UInt8" Format="HDF">
                    {hdf5_rel_path}:/Mesh/Level
                </DataItem>
            </Attribute>
        </Grid>
    </Domain>
</Xdmf>
"""
    with open(xdmf_filename, 'w') as f:
        f.write(xdmf_content)
    print("    XDMF file written successfully")

# Voxelize an STL file using Open3D
def voxelize_stl_open3d(stl_filename, length_lbm_unit):
    tic = time.perf_counter()
    mesh = o3d.io.read_triangle_mesh(stl_filename)
    if mesh.is_empty():
        raise ValueError("The mesh is empty or invalid.")
    if length_lbm_unit <= 0:
        raise ValueError("Voxel size must be a positive number.")
    print(f"    Number of vertices: {len(mesh.vertices):,}")
    print(f"    Number of triangles: {len(mesh.triangles):,}")
    toc = time.perf_counter()
    print(f"    Model read in {toc - tic:0.1f} seconds")
    tic = time.perf_counter()
    min_bound = mesh.get_axis_aligned_bounding_box().get_min_bound()
    max_bound = mesh.get_axis_aligned_bounding_box().get_max_bound()
    min_bound = np.floor(min_bound / length_lbm_unit) * length_lbm_unit
    max_bound = np.ceil(max_bound / length_lbm_unit) * length_lbm_unit
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=length_lbm_unit, min_bound=min_bound, max_bound=max_bound
    )
    bbox = voxel_grid.get_axis_aligned_bounding_box()
    grid_size = np.ceil((bbox.get_max_bound() - bbox.get_min_bound()) / length_lbm_unit).astype(int)
    voxel_matrix = np.zeros(grid_size, dtype=bool)
    voxel_indices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    voxel_matrix[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = True
    origin = bbox.get_box_points()[0]
    toc = time.perf_counter()
    print(f"    Grid created in {toc - tic:0.1f} seconds")
    return voxel_matrix, origin

# Save a NumPy array to VTK as cell data
def save_numpy_to_vtk_as_cell_data(array, filename, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    if array.ndim != 3:
        raise ValueError("Array must be 3D")
    nx, ny, nz = array.shape
    vtk_data_array = numpy_to_vtk(array.ravel('F'), deep=True, array_type=vtk.VTK_FLOAT)
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(nx + 1, ny + 1, nz + 1)
    image_data.SetSpacing(*spacing)
    image_data.SetOrigin(*origin)
    image_data.GetCellData().SetScalars(vtk_data_array)
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()

# Save a NumPy array to VTK as point data
def save_numpy_to_vtk(array, filename, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    if array.ndim != 3:
        raise ValueError("Array must be 3D")
    nx, ny, nz = array.shape
    vtk_data_array = numpy_to_vtk(array.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(nx, ny, nz)
    image_data.SetSpacing(*spacing)
    image_data.SetOrigin(*origin)
    image_data.GetPointData().SetScalars(vtk_data_array)
    writer = vtk.vtkStructuredPointsWriter()
    writer.SetFileName(filename)
    writer.SetInputData(image_data)
    writer.Write()

# Print a table of padding values for each level
def print_padding_table(padding_values):
    headers = ["Level", "X-", "X+", "Y-", "Y+", "Z-", "Z+"]
    table = [[level] + list(values) for level, values in padding_values.items()]
    print(tabulate(table, headers=headers, tablefmt="grid"))

# Generate a multi-level voxel mesh from an STL file
def makeMesh(levels, filename, voxSize, kernel, domainMin, domainMax, close=True, ground_refinement_level=None):
    stem = Path(filename).stem
    tic = time.perf_counter()
    maxVoxSize = voxSize * pow(2, levels - 1)
    domainMin = np.round(domainMin / maxVoxSize) * maxVoxSize
    domainMax = np.round(domainMax / maxVoxSize) * maxVoxSize
    print("\n" + "=" * 100 + "\n")
    print("Meshing Configuration:")
    print(f"Model: {filename}")
    print(f"Finest level: {voxSize} meters")
    print(f"Number of levels: {levels}")
    print(f"Close islands: {close}")
    if ground_refinement_level is not None:
        print(f"Ground refinement level: {ground_refinement_level}")
    print("Adjusted domain coordinates: ", domainMin, ", ", domainMax)
    print("Voxel growth strategy:")
    print_padding_table(padding_values)
    print("\n" + "=" * 100 + "\n")

    domainSize = (domainMax - domainMin)
    print("/// Make Mesh started... " + stem)
    v = voxSize

    level_data = []
    matrix, origin = voxelize_stl_open3d(filename, voxSize)
    print("/// Level 0 voxel size: ", v)
    ticLevel = time.perf_counter()
    g, origin = grow_gpu(matrix, voxSize, origin, kernel[0])
    f, origin = fill_gpu(g, voxSize, origin, close)
    df, origin = crop_gpu(f, origin, domainMin, domainMax, v)
    dOrigin = np.copy(origin)
    dr = df.copy()  # dr is the final matrix for this level
    level_data.append((dr, v, dOrigin, 0))  # Store only dr
    tocLevel = time.perf_counter()
    print(f"    Level defined in {tocLevel - ticLevel:0.1f} seconds")
    
    ground_z = domainMin[2]

    for l in range(1, levels):
        ticLevel = time.perf_counter()
        d = df[::2, ::2, ::2]
        v = v * 2
        print("/// Level", l, "voxel size:", v)
        full_shape = np.round(domainSize / v).astype(int)

        if l < levels - 1:
            dg, dOrigin = grow_gpu(d, v, dOrigin, kernel[l])
            df_natural, dOrigin = fill_gpu(dg, v, dOrigin, close)
            df_natural, dOrigin = crop_gpu(df_natural, dOrigin, domainMin, domainMax, v)
        else:
            df_natural = np.ones(tuple(full_shape), bool)
            dOrigin = domainMin

        if ground_refinement_level is not None and l == ground_refinement_level:
            df_ground = np.zeros(tuple(full_shape), bool)
            dOrigin_ground = domainMin
            ground_z_index = int(np.round((ground_z - dOrigin_ground[2]) / v))
            n_thick = 4
            if 0 <= ground_z_index < full_shape[2]:
                end_z = min(ground_z_index + n_thick, full_shape[2])
                df_ground[:, :, ground_z_index:end_z] = True

            offset = np.round((dOrigin - dOrigin_ground) / v).astype(int)
            x0, y0, z0 = offset
            x1, y1, z1 = x0 + df_natural.shape[0], y0 + df_natural.shape[1], z0 + df_natural.shape[2]
            x0, y0, z0 = np.maximum([x0, y0, z0], 0)
            x1, y1, z1 = np.minimum([x1, y1, z1], full_shape)
            df = np.zeros(tuple(full_shape), bool)
            if x1 > x0 and y1 > y0 and z1 > z0:
                df[x0:x1, y0:y1, z0:z1] = df_natural[:(x1 - x0), :(y1 - y0), :(z1 - z0)]
            df |= df_ground
            dOrigin = dOrigin_ground
        else:
            df = df_natural

        dr = remove_gpu(df, dOrigin, d, origin, v)
        level_data.append((dr, v, dOrigin, l))  # Store only dr
        tocLevel = time.perf_counter()
        print(f"    Level defined in {tocLevel - ticLevel:0.1f} seconds")
        origin = np.copy(dOrigin)

    toc = time.perf_counter()
    print(f"    Generated mesh in {toc - tic:0.1f} seconds")

    print("/// Mesh Data Report")
    finest_possible_voxels = int(np.prod(domainSize / voxSize))
    total_voxels_billions = finest_possible_voxels / 1e9
    print(f"    Total domain size: {total_voxels_billions:.2f} billion voxels (if filled at finest resolution {voxSize} m)")

    total_voxel_count = sum(np.sum(dr) for dr, _, _, _ in level_data)
    total_voxel_count_millions = total_voxel_count / 1e6
    print(f"    Total voxel count: {total_voxel_count_millions:.2f} million")

    percentage_reduction = ((finest_possible_voxels - total_voxel_count) / finest_possible_voxels) * 100 if finest_possible_voxels > 0 else 0
    print(f"    Percentage reduction: {percentage_reduction:.2f}% (vs. uniform dense grid)")

    print("    Voxel distribution per level:")
    headers = ["Level", "Voxel Size (m)", "Voxels (M)", "Percentage (%)"]
    table_data = []
    for l, (dr, v, _, _) in enumerate(level_data):
        voxel_count = np.sum(dr)
        voxel_count_millions = voxel_count / 1e6
        percentage = (voxel_count / total_voxel_count) * 100 if total_voxel_count > 0 else 0
        table_data.append([l, v, f"{voxel_count_millions:.2f}", f"{percentage:.2f}"])
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()

    # File output
    ### VTU Output ###
    save_to_unstructured_vtk(
        [(dr, v, dOrigin, l) for dr, v, dOrigin, l in level_data],
        stem + "_Unstructured_WARP.vtu", voxSize)

    ### HDF5 Output ###
    total_cells, num_points = save_to_hdf5(
        [(dr, v, dOrigin, l) for dr, v, dOrigin, l in level_data],
        stem + "_Unstructured_WARP.h5", voxSize)
    save_xdmf(stem + "_Unstructured_WARP.h5", stem + "_Unstructured_WARP.xmf", total_cells, num_points)

    print("    MakeMesh Completed!")
    print()

# Calculate convolution kernels based on padding values
def calculate_kernel(padding_values):
    kernels = {}
    for level, (xn, xp, yn, yp, zn, zp) in padding_values.items():
        x_dim = max(xp, xn) * 2 + 1
        y_dim = max(yp, yn) * 2 + 1
        z_dim = max(zp, zn) * 2 + 1
        ones_x = xp + xn + 1
        ones_y = yp + yn + 1
        ones_z = zp + zn + 1
        mid_x = (x_dim - 1) // 2
        x1 = mid_x - xp
        x2 = x_dim - (mid_x - xn)
        mid_y = (y_dim - 1) // 2
        y1 = mid_y - yp
        y2 = y_dim - (mid_y - yn)
        mid_z = (z_dim - 1) // 2
        z1 = mid_z - zp
        z2 = z_dim - (mid_z - zn)
        kernels[level] = np.zeros((x_dim, y_dim, z_dim), bool)
        kernels[level][x1:x2, y1:y2, z1:z2] = np.ones((x2 - x1, y2 - y1, z2 - z1), bool)
    return kernels

# Values to be used for Studio Wind Tunnel
padding_values = {
    0: (2, 2, 2, 2, 2, 2),
    1: (4, 4, 4, 4, 4, 4),
    2: (4, 40, 8, 8, 8, 8),
    3: (4, 40, 8, 8, 8, 8),
    4: (4, 40, 4, 4, 4, 4),
    5: (4, 40, 4, 4, 4, 4),
    6: (4, 4, 4, 4, 4, 4),
    7: (4, 4, 4, 4, 4, 4),
    8: (4, 4, 4, 4, 4, 4),
}

# Values to be used for Lid Driven Cavity
# padding_values = {
    # 0: (2, 2, 2, 2, 2, 2),
    # 1: (4, 4, 4, 4, 4, 4),
    # 2: (4, 4, 4, 4, 4, 4),
    # 3: (4, 4, 4, 4, 4, 4),
    # 4: (4, 4, 4, 4, 4, 4),
    # 5: (4, 4, 4, 4, 4, 4),
    # 6: (4, 4, 4, 4, 4, 4),
    # 7: (4, 4, 4, 4, 4, 4),
# }

kernel = calculate_kernel(padding_values)

# Execute the meshing process
makeMesh(8, "Ahmed_25.stl", 0.001, kernel, np.array([0, -5, 0], float), np.array([10, 5, 5], float), True, ground_refinement_level=4)

#makeMesh(7,"UnitCube.stl",0.00195694716242661448140900195695,kernel,np.array([0,0,0],float),np.array([1,1,1],float),True)

#makeMesh(7,"S550_GT500_BS_Combined.stl",0.003,kernel,np.array([-10,-10,0],float),np.array([25,10,10],float),True)