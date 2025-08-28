# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
import numpy as np
import trimesh
import os
import sys
from loguru import logger
from torch.utils.cpp_extension import load

face_renderer = None
def set_rasterizer(name = 'standard'):
    global face_renderer
    if name == 'pytorch3d':
        logger.error('pytorch3d not supported, use standard instead')
        raise RuntimeError('pytorch3d not supported, use standard instead')
    elif name == 'standard':
        if face_renderer is None:
            standard_rasterize_cpu = load(name='standard_rasterize_cpu',
                 sources=[
                     'decalib/utils/rasterizer/standard_rasterize_cpu.cpp',
                     'decalib/utils/rasterizer/standard_rasterize_cpu_kernel.cc'
                 ],
                 with_cuda=False,
                 verbose=False)
            face_renderer = standard_rasterize_cpu
            logger.info('set rasterizer to standard')
        else:
            logger.info('rasterizer is already standard')
    else:
        logger.error('rasterizer not supported')
        raise RuntimeError('rasterizer not supported')

def get_rasterizer():
    global face_renderer
    return face_renderer

# Helper functions to load mesh data using trimesh
def load_mesh_data(obj_filename):
    mesh = trimesh.load(obj_filename)
    # The original code might be expecting texture coordinates (vt) and faces (ft)
    vt = mesh.vertices[mesh.faces.flatten()]
    ft = mesh.faces
    return vt, ft

def load_faces(obj_filename):
    mesh = trimesh.load(obj_filename)
    return mesh.faces

class SRenderY(nn.Module):
    def __init__(self, size=256, obj_filename='models/mean.obj', uv_size=256, rasterizer_type='standard'):
        super(SRenderY, self).__init__()
        self.rasterizer_type = rasterizer_type
        if rasterizer_type == 'standard':
            self.rasterizer = get_rasterizer()
            
            # Use trimesh to load mesh data, as the C++ extension does not have a load_vt function.
            vt, ft = load_mesh_data(obj_filename)
            faces = load_faces(obj_filename)
            
            # Register buffers directly to prevent a KeyError.
            self.register_buffer('raw_uvcoords', torch.from_numpy(vt).float()[None, :])
            self.register_buffer('uvfaces', torch.from_numpy(ft).long()[None, :])
            
        elif rasterizer_type == 'pytorch3d':
            logger.error('pytorch3d not supported, use standard instead')
            raise RuntimeError('pytorch3d not supported, use standard instead')
        else:
            logger.error('rasterizer not supported')
            raise RuntimeError('rasterizer not supported')

        # Get the buffer values from the registered buffers instead of reassigning them.
        self.uvcoords = self.raw_uvcoords.clone()
        self.uvfaces = self.uvfaces.clone()
        
        self.register_buffer('faces', torch.from_numpy(faces).long()[None, :])
        self.image_size = size
        self.uv_size = uv_size
        self.faces = self.faces.long()
        self.uvfaces = self.uvfaces.long()
        self.verts_uvs = self.raw_uvcoords
        self.faces_uvs = self.uvfaces
        self.n_face = self.faces.shape[1]
        self.n_uv_face = self.uvfaces.shape[1]
        
        self.mean_tex = np.ones((self.uv_size, self.uv_size, 3)).astype(np.float32)

    def to(self, device):
        self.device = device
        self.faces = self.faces.to(device)
        self.uvfaces = self.uvfaces.to(device)
        self.uvcoords = self.uvcoords.to(device)
        return self

    def forward(self, vertices, transformed_vertices, textures, h=None, w=None, background=None):
        if self.rasterizer_type == 'pytorch3d':
            logger.error('pytorch3d not supported, use standard instead')
            raise RuntimeError('pytorch3d not supported, use standard instead')
        
        if h is None:
            h = self.image_size
        if w is None:
            w = self.image_size
            
        # Allocate the required buffers before calling the C++ function
        batch_size = vertices.shape[0]
        depth_buffer = -torch.ones([batch_size, h, w], dtype=torch.float32, device=self.device)
        triangle_buffer = -torch.ones([batch_size, h, w], dtype=torch.int32, device=self.device)
        
        # --- CHANGE START ---
        # Explicitly convert the faces tensor to Float type before passing it to the C++ function
        faces_float = self.faces.float()
        image, alpha, face_ids = self.rasterizer.rasterize(
            transformed_vertices, 
            faces_float, 
            depth_buffer, 
            triangle_buffer, 
            h, w
        )
        # --- CHANGE END ---
        
        uvcoords = self.raw_uvcoords.repeat(vertices.shape[0], 1, 1)
        uvfaces = self.uvfaces.repeat(vertices.shape[0], 1, 1)
        
        attributes = self.rasterizer.create_attributes(vertices, self.faces, uvcoords, uvfaces, textures)
        rendered_images = self.rasterizer.interpolate(image, attributes, alpha, h, w)
        rendered_images = rendered_images.permute(0, 3, 1, 2)
        alpha = alpha.permute(0, 3, 1, 2)
        
        return {
            'images': rendered_images,
            'alpha_images': alpha,
            'face_ids': face_ids,
            'grid': image
        }

    def render_uv(self, vertices, uvcoords, textures, h=None, w=None):
        if h is None:
            h = self.uv_size
        if w is None:
            w = self.uv_size
        uv_vertices = torch.cat([uvcoords, torch.zeros_like(uvcoords[:,:,:1])], dim=-1)
        image, alpha, face_ids = self.rasterizer.rasterize(uv_vertices, self.uvfaces, h, w)
        
        attributes = self.rasterizer.create_attributes(vertices, self.faces, uvcoords, self.uvfaces, textures, is_uv=True)
        rendered_images = self.rasterizer.interpolate(image, attributes, alpha, h, w)
        rendered_images = rendered_images.permute(0, 3, 1, 2)
        alpha = alpha.permute(0, 3, 1, 2)

        return {
            'images': rendered_images,
            'alpha_images': alpha,
            'face_ids': face_ids,
            'grid': image
        }

    def world2uv(self, vertices, h=None, w=None):
        if h is None:
            h = self.uv_size
        if w is None:
            w = self.uv_size
        uv_vertices = torch.cat([self.uvcoords, torch.zeros_like(self.uvcoords[:, :, :1])], dim=-1)
        image, alpha, face_ids = self.rasterizer.rasterize(uv_vertices, self.uvfaces, h, w)
        
        attributes = self.rasterizer.create_attributes(vertices, self.faces, self.uvcoords, self.uvfaces)
        uv_verts_vis = self.rasterizer.interpolate(image, attributes, alpha, h, w)
        uv_verts_vis = uv_verts_vis.permute(0, 3, 1, 2)
        
        return uv_verts_vis
        
    def add_SHlight(self, normal_images, light_coef):
        n = normal_images
        sh_basis = torch.stack([
            n[:,0] * 0.5/np.pi, 
            n[:,1] * 0.5/np.pi, 
            n[:,2] * 0.5/np.pi,
            (n[:,0] * n[:,1]) * 1/np.pi,
            (n[:,1] * n[:,2]) * 1/np.pi,
            (n[:,2] * n[:,0]) * 1/np.pi,
            (n[:,0]**2 - n[:,1]**2) * 1/np.pi,
            (3 * n[:,2]**2 - 1) * 1/(2*np.pi),
            (n[:,0]**2 + n[:,1]**2 + n[:,2]**2) * 1/(4*np.pi) - 1/(4*np.pi)
        ], 1)
        sh_basis = sh_basis*self.uv_face_mask
        
        sh_shading = torch.einsum('bcehw, bdc->bdhw', sh_basis.unsqueeze(2), light_coef)
        
        return sh_shading


    def render_depth(self, transformed_vertices, h=None, w=None):
        if h is None:
            h = self.image_size
        if w is None:
            w = self.image_size
        depth_images, alpha, face_ids = self.rasterizer.rasterize(transformed_vertices, self.faces, h, w)
        depth_images = depth_images.permute(0, 3, 1, 2)
        return depth_images

    def render_shape(self, vertices, transformed_vertices, h=None, w=None, images=None, return_grid=False, detail_normal_images=None):
        if h is None:
            h = self.image_size
        if w is None:
            w = self.image_size
        
        shape_images, alpha, face_ids = self.rasterizer.rasterize(transformed_vertices, self.faces, h, w)
        grid = shape_images.clone()
        
        if detail_normal_images is None:
            normals = util.vertex_normals(vertices, self.faces.expand(vertices.shape[0], -1, -1))
            attributes = self.rasterizer.create_attributes(vertices, self.faces, self.uvcoords, self.uvfaces, normals, is_normal=True)
            normal_images = self.rasterizer.interpolate(shape_images, attributes, alpha, h, w)
            normal_images = normal_images.permute(0, 3, 1, 2)
            
            shading = normal_images[:,:3,:,:]*0.5 + 0.5
            shape_images = shading
            shape_images = shape_images * alpha + (1-alpha) * torch.ones_like(shape_images)
        else:
            shading = detail_normal_images[:,:3,:,:]*0.5 + 0.5
            shape_images = shading * alpha + (1-alpha) * torch.ones_like(shading)
        
        shape_images = shape_images[:,:,:,[2,1,0]]
        
        if return_grid:
            return shape_images, alpha, grid, None
        else:
            return shape_images, alpha
