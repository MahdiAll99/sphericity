#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from copy import deepcopy
from typing import List

import numpy as np

from ..utils.image_volume_obj import image_volume_obj
from ..utils.imref import imref3d, intrinsicToWorld, worldToIntrinsic
from ..utils.interp3 import interp3


def interp_volume(
        vol_obj_s: image_volume_obj,
        vox_dim: List,
        interp_met: str,
        round_val: float,
        image_type: str) -> image_volume_obj:
    """3D voxel interpolation on the input volume.

    Args:
        vol_obj_s (image_volume_obj): Imaging  that will be interpolated.
        medscan (object): The MEDscan class object.
        vox_dim (array): Array of the voxel dimension. The following format is used
                         [Xin,Yin,Zslice], where Xin and Yin are the X (left to right) and
                         Y (bottom to top) IN-PLANE resolutions, and Zslice is the slice spacing,
                         no matter the orientation of the volume (i.e. axial , sagittal, coronal).
        interp_met (str): {nearest, linear, spline, cubic} optional, Interpolation method
        round_val (float): Rounding value. Must be between 0 and 1 for ROI interpolation
                           and to a power of 10 for Image interpolation.
        image_type (str): 'image' for imaging data interpolation and 'roi' for ROI mask data interpolation.
        roi_obj_s (image_volume_obj): Mask data, will be used to compute a new specific box
                                      and the new imref3d object for the imaging data.
        box_string (str): Specifies the size if the box containing the ROI

                          - 'full': full imaging data as output.
                          - 'box': computes the smallest bounding box.
                          - Ex: 'box10': 10 voxels in all three dimensions are added to \
                            the smallest bounding box. The number after 'box' defines the \
                            number of voxels to add.
                          - Ex: '2box': Computes the smallest box and outputs double its \
                            size. The number before 'box' defines the multiplication in size.
        texture (bool): If True, the texture voxel spacing of ``MEDscan`` will be used for interpolation.
    
    Returns:
        ndarray: 3D array of 1's and 0's defining the ROI mask.
    """
    try:
        # PARSING ARGUMENTS
        if np.sum(vox_dim) == 0:
            return deepcopy(vol_obj_s)
        if len(vox_dim) == 2:
            two_d = True
        else:
            two_d = False
                
        if image_type is None:
            raise ValueError(
                "The type of input image should be specified as \"image\" or \"roi\".")
        elif image_type not in ["image", "roi"]:
            raise ValueError(
                "The type of input image should either be \"image\" or \"roi\".")
        elif image_type == "image":
            if interp_met not in ["linear", "cubic", "spline"]:
                raise ValueError(
                    "Interpolation method for images should either be \"linear\", \"cubic\" or \"spline\".")
            if round_val is not None:
                if np.mod(np.log10(round_val), 1):
                    raise ValueError("\"round_val\" should be a power of 10.")
        else:
            if interp_met not in ["nearest", "linear", "cubic"]:
                raise ValueError(
                    "Interpolation method for images should either be \"nearest\", \"linear\" or \"cubic\".")
            if round_val is not None:
                if round_val < 0.0 or round_val > 1.0:
                    raise ValueError("\"round_val\" must be between 0.0 and 1.0.")
            else:
                raise ValueError("\"round_val\" must be provided for \"roi\".")

        # --> QUERIED POINTS: NEW INTERPOLATED VOLUME: "q" or "Q".
        # --> SAMPLED POINTS: ORIGINAL VOLUME: "s" or "S".
        # --> Always using XYZ coordinates (unless specifically noted),
        #     not MATLAB IJK, so beware!

        # INITIALIZATION
        res_q = vox_dim
        if two_d:
            # If 2D, the resolution of the slice dimension of he queried volume is
            # set to the same as the sampled volume.
            res_q = np.concatenate((res_q, vol_obj_s.spatialRef.PixelExtentInWorldZ))

        res_s = np.array([vol_obj_s.spatialRef.PixelExtentInWorldX,
                        vol_obj_s.spatialRef.PixelExtentInWorldY,
                        vol_obj_s.spatialRef.PixelExtentInWorldZ])

        if np.array_equal(res_s, res_q):
            return deepcopy(vol_obj_s)

        spatial_ref_s = vol_obj_s.spatialRef
        extent_s = np.array([spatial_ref_s.ImageExtentInWorldX,
                            spatial_ref_s.ImageExtentInWorldY,
                            spatial_ref_s.ImageExtentInWorldZ])
        low_limits_s = np.array([spatial_ref_s.XWorldLimits[0],
                            spatial_ref_s.YWorldLimits[0],
                            spatial_ref_s.ZWorldLimits[0]])

        # CREATING QUERIED "imref3d" OBJECT CENTERED ON SAMPLED VOLUME

        # Switching to IJK (matlab) reference frame for "imref3d" computation.
        # Putting a "ceil", according to IBSI standards. This is safer than "round".
        size_q = np.ceil(np.around(np.divide(extent_s, res_q),
                                decimals=3)).astype(int).tolist()

        if two_d:
            # If 2D, forcing the size of the queried volume in the slice dimension
            # to be the same as the sample volume.
            size_q[2] = vol_obj_s.spatialRef.ImageSize[2]

        spatial_ref_q = imref3d(imageSize=size_q, 
                            pixelExtentInWorldX=res_q[0],
                            pixelExtentInWorldY=res_q[1],
                            pixelExtentInWorldZ=res_q[2])

        extent_q = np.array([spatial_ref_q.ImageExtentInWorldX,
                            spatial_ref_q.ImageExtentInWorldY,
                            spatial_ref_q.ImageExtentInWorldZ])
        low_limits_q = np.array([spatial_ref_q.XWorldLimits[0],
                            spatial_ref_q.YWorldLimits[0],
                            spatial_ref_q.ZWorldLimits[0]])
        diff = extent_q - extent_s
        new_low_limits_q = low_limits_s - diff/2
        spatial_ref_q.XWorldLimits = spatial_ref_q.XWorldLimits - \
            (low_limits_q[0] - new_low_limits_q[0])
        spatial_ref_q.YWorldLimits = spatial_ref_q.YWorldLimits - \
            (low_limits_q[1] - new_low_limits_q[1])
        spatial_ref_q.ZWorldLimits = spatial_ref_q.ZWorldLimits - \
            (low_limits_q[2] - new_low_limits_q[2])

        # CREATING QUERIED XYZ POINTS
        x_q = np.arange(size_q[0])
        y_q = np.arange(size_q[1])
        z_q = np.arange(size_q[2])
        x_q, y_q, z_q = np.meshgrid(x_q, y_q, z_q, indexing='ij')
        x_q, y_q, z_q = intrinsicToWorld(
            R=spatial_ref_q, xIntrinsic=x_q, yIntrinsic=y_q, zIntrinsic=z_q)

        # CONVERTING QUERIED XZY POINTS TO INTRINSIC COORDINATES IN THE SAMPLED
        # REFERENCE FRAME
        x_q, y_q, z_q = worldToIntrinsic(
            R=spatial_ref_s, xWorld=x_q, yWorld=y_q, zWorld=z_q)

        # INTERPOLATING VOLUME
        data = interp3(v=vol_obj_s.data, x_q=x_q, y_q=y_q, z_q=z_q, method=interp_met)
        vol_obj_q = image_volume_obj(data=data, spatial_ref=spatial_ref_q)

        # ROUNDING
        if image_type == "image":
            # Grey level rounding for "image" type
            if round_val is not None and (type(round_val) is int or type(round_val) is float):
                # DELETE NEXT LINE WHEN THE RADIOMICS PARAMETER OPTIONS OF
                # interp.glRound ARE FIXED
                round_val = (-np.log10(round_val)).astype(int)
                vol_obj_q.data = np.around(vol_obj_q.data, decimals=round_val)
        else:
            vol_obj_q.data[vol_obj_q.data >= round_val] = 1.0
            vol_obj_q.data[vol_obj_q.data < round_val] = 0.0

    except Exception as e:
        print(f"\n PROBLEM WITH INTERPOLATION:\n {e}")

    return vol_obj_q
