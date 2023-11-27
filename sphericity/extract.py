from copy import deepcopy
from typing import List, Tuple, Union
import numpy as np
import nibabel as nib
from skimage.measure import marching_cubes
from pathlib import Path

# Init processing & computation parameters
from . import processing
from . import utils


def associate_spatialRef(nifti_file: Union[Path, str]) -> utils.imref3d:
    """Computes the imref3d spatialRef using a NIFTI file and updates the spatialRef attribute.

    Args:
        nifti_file(Union[Path, str]): Path to the nifti data.

    Returns:
        MEDimage: Returns a MEDimage instance with updated spatialRef attribute.
    """
    # Loading the nifti file :
    nifti = nib.load(nifti_file)
    nifti_data = nib.load(nifti_file).get_fdata()

    # spatialRef Creation
    pixel_x = abs(nifti.affine[0, 0])
    pixel_y = abs(nifti.affine[1, 1])
    slices = abs(nifti.affine[2, 2])
    min_grid = nifti.affine[:3, 3]
    min_x_grid = min_grid[0]
    min_y_grid = min_grid[1]
    min_z_grid = min_grid[2]
    size_image = np.shape(nifti_data)
    spatialRef = utils.imref3d(size_image, abs(pixel_x), abs(pixel_y), abs(slices))
    spatialRef.XWorldLimits = (np.array(spatialRef.XWorldLimits) - (spatialRef.XWorldLimits[0] -(min_x_grid-pixel_x/2))).tolist()
    spatialRef.YWorldLimits = (np.array(spatialRef.YWorldLimits) - (spatialRef.YWorldLimits[0] -(min_y_grid-pixel_y/2))).tolist()
    spatialRef.ZWorldLimits = (np.array(spatialRef.ZWorldLimits) - (spatialRef.ZWorldLimits[0] -(min_z_grid-slices/2))).tolist()

    # Converting the results into lists
    spatialRef.ImageSize = spatialRef.ImageSize.tolist()
    spatialRef.XIntrinsicLimits = spatialRef.XIntrinsicLimits.tolist()
    spatialRef.YIntrinsicLimits = spatialRef.YIntrinsicLimits.tolist()
    spatialRef.ZIntrinsicLimits = spatialRef.ZIntrinsicLimits.tolist()

    return spatialRef

def get_mesh(
        mask: np.ndarray,
        res: Union[np.ndarray, List]
    ) -> Tuple[np.ndarray,
            np.ndarray,
            np.ndarray]:
    """Compute Mesh.

    Note:
      Make sure the `mask` is padded with a layer of 0's in all
      dimensions to reduce potential isosurface computation errors.

    Args:
        mask (ndarray): Contains only 0's and 1's.
        res (ndarray or List): [a,b,c] vector specifying the resolution of the volume in mm.
            xyz resolution (world), or JIK resolution (intrinsic matlab).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - Array of the [X,Y,Z] positions of the ROI.
            - Array of the spatial coordinates for `mask` unique mesh vertices.
            - Array of triangular faces via referencing vertex indices from vertices.
    """
    # Getting the grid of X,Y,Z positions, where the coordinate reference
    # system (0,0,0) is located at the upper left corner of the first voxel
    # (-0.5: half a voxel distance). For the whole volume defining the mask,
    # no matter if it is a 1 or a 0.
    mask = mask.copy()
    res = res.copy()

    x = res[0]*((np.arange(1, np.shape(mask)[0]+1))-0.5)
    y = res[1]*((np.arange(1, np.shape(mask)[1]+1))-0.5)
    z = res[2]*((np.arange(1, np.shape(mask)[2]+1))-0.5)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Getting the isosurface of the mask
    vertices, faces, _, _ = marching_cubes(volume=mask, level=0.5, spacing=res)

    # Getting the X,Y,Z positions of the ROI (i.e. 1's) of the mask
    X = np.reshape(X, (np.size(X), 1), order='F')
    Y = np.reshape(Y, (np.size(Y), 1), order='F')
    Z = np.reshape(Z, (np.size(Z), 1), order='F')

    xyz = np.concatenate((X, Y, Z), axis=1)
    xyz = xyz[np.where(np.reshape(mask, np.size(mask), order='F') == 1)[0], :]

    return xyz, faces, vertices

def padding(mask_morph: np.ndarray) -> np.ndarray:
    """Padding the volume and masks.

    Args:
        mask_morph (ndarray): Morphological mask.

    Returns:
        tuple of 3 ndarray: Volume and masks after padding.
    """
    # PADDING THE MASKS WITH A LAYER OF 0's
    # (reduce mesh computation errors of associated mask)
    mask_morph = mask_morph.copy()
    mask_morph = np.pad(mask_morph, pad_width=1, mode="constant", constant_values=0.0)

    return mask_morph

def get_mesh_volume(
        faces: np.ndarray,
        vertices:np.ndarray
    ) -> float:
    """Computes MeshVolume feature.
    This feature refers to "Volume (mesh)" (ID = RNU0)  
    in the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        faces (np.ndarray): matrix of three column vectors, defining the [X,Y,Z]
                          positions of the ``faces`` of the isosurface or convex hull of the mask
                          (output from "isosurface.m" or "convhull.m" functions of MATLAB).
                          --> These are more precisely indexes to ``vertices``
        vertices (np.ndarray): matrix of three column vectors, defining the
                             [X,Y,Z] positions of the ``vertices`` of the isosurface of the mask (output
                             from "isosurface.m" function of MATLAB).
                             --> In mm.

    Returns:
        float: Mesh volume
    """
    faces = faces.copy()
    vertices = vertices.copy()

    # Getting vectors for the three vertices
    # (with respect to origin) of each face
    a = vertices[faces[:, 0], :]
    b = vertices[faces[:, 1], :]
    c = vertices[faces[:, 2], :]

    # Calculating volume
    v_cross = np.cross(b, c)
    v_dot = np.sum(a.conj()*v_cross, axis=1)
    volume = np.abs(np.sum(v_dot))/6

    return volume

def get_mesh_area(
        faces: np.ndarray,
        vertices: np.ndarray
    ) -> float:
    """Computes the surface area (mesh) feature from the ROI mesh by 
    summing over the triangular face surface areas. 
    This feature refers to "Surface area (mesh)" (ID = C0JK)  
    in the `IBSI1 reference manual <https://arxiv.org/pdf/1612.07003.pdf>`_.

    Args:
        faces (np.ndarray): matrix of three column vectors, defining the [X,Y,Z]
                          positions of the ``faces`` of the isosurface or convex hull of the mask
                          (output from "isosurface.m" or "convhull.m" functions of MATLAB).
                          --> These are more precisely indexes to ``vertices``
        vertices (np.ndarray): matrix of three column vectors,
                             defining the [X,Y,Z]
                             positions of the ``vertices`` of the isosurface of the mask (output
                             from "isosurface.m" function of MATLAB).
                             --> In mm.

    Returns:
        float: Mesh area.
    """

    faces = faces.copy()
    vertices = vertices.copy()

    # Getting two vectors of edges for each face
    a = vertices[faces[:, 1], :] - vertices[faces[:, 0], :]
    b = vertices[faces[:, 2], :] - vertices[faces[:, 0], :]

    # Calculating the surface area of each face and summing it up all at once.
    c = np.cross(a, b)
    area = 1/2 * np.sum(np.sqrt(np.sum(np.power(c, 2), 1)))

    return area

def extract(
        path_nifti_mask: Union[str, Path],
        voxel_dim: List,
        roi_interp: str,
        roi_pv: float
    ) -> float:
    """
    Computes the Spherecity (IBSI code: QCFX) feature from the given data.

    Args:
        path_nifti_mask (Union[str, Path]): Path to the nifti mask.
        voxel_dim (List): List of resampled voxel spacing (mm).
        roi_interp (str): Interpolation method used for the ROI. Must be "nearest", "linear" or "cubic".
        roi_pv (float): ROI partial mask volume (ie. Rounding value for the ROI mask in interpolation).

    Returns:
        float: Sphericity feature.
    """
    # Mask - Loading
    roi_data = nib.load(path_nifti_mask).get_fdata()

    # SpatialRef creation
    spatial_ref = associate_spatialRef(path_nifti_mask)

    # Convert to Image Volume Object
    roi_obj_init = utils.image_volume_obj(roi_data, spatial_ref)

    # Interpolation
    # Morphological Mask
    roi_obj_morph = processing.interp_volume(
        vol_obj_s=roi_obj_init,
        vox_dim=voxel_dim,
        interp_met=roi_interp,
        round_val=roi_pv,
        image_type='roi'
    )

    # Initialization
    mask_morph = roi_obj_morph.data

    # Padding the mask
    mask_morph = padding(mask_morph)

    # Mesh volume and area
    _, faces, vertices = get_mesh(mask_morph, voxel_dim)
    volume = get_mesh_volume(faces, vertices)
    area = get_mesh_area(faces, vertices)

    # Sphericity
    return ((36*np.pi*volume**2)**(1/3)) / area
