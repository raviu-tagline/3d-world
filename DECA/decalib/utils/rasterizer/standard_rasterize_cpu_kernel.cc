// Ref: https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/cuda/rasterize_cuda_kernel.cu
// https://github.com/YadiraF/face3d/blob/master/face3d/mesh/cython/mesh_core.cpp

#include <algorithm>
#include <cmath>
#include <vector>

#include <ATen/ATen.h>

template <typename scalar_t>
inline bool check_face_frontside(const scalar_t* face) {
    return (face[7] - face[1]) * (face[3] - face[0]) < (face[4] - face[1]) * (face[6] - face[0]);
}

template <typename scalar_t> struct point {
public:
    scalar_t x;
    scalar_t y;

    scalar_t dot(point<scalar_t> p) {
        return this->x * p.x + this->y * p.y;
    };

    point<scalar_t> operator-(point<scalar_t>& p) {
        point<scalar_t> np;
        np.x = this->x - p.x;
        np.y = this->y - p.y;
        return np;
    };

    point<scalar_t> operator+(point<scalar_t>& p) {
        point<scalar_t> np;
        np.x = this->x + p.x;
        np.y = this->y + p.y;
        return np;
    };

    point<scalar_t> operator*(scalar_t s) {
        point<scalar_t> np;
        np.x = s * this->x;
        np.y = s * this->y;
        return np;
    };
};

template <typename scalar_t>
inline bool check_pixel_inside(const scalar_t* w) {
    return w[0] <= 1 && w[0] >= 0 && w[1] <= 1 && w[1] >= 0 && w[2] <= 1 && w[2] >= 0;
}

template <typename scalar_t>
inline void barycentric_weight(scalar_t* w, point<scalar_t> p, point<scalar_t> p0, point<scalar_t> p1, point<scalar_t> p2) {
    point<scalar_t> v0, v1, v2;
    v0 = p2 - p0;
    v1 = p1 - p0;
    v2 = p - p0;

    scalar_t dot00 = v0.dot(v0);
    scalar_t dot01 = v0.dot(v1);
    scalar_t dot02 = v0.dot(v2);
    scalar_t dot11 = v1.dot(v1);
    scalar_t dot12 = v1.dot(v2);

    scalar_t inverDeno;
    if (dot00 * dot11 - dot01 * dot01 == 0)
        inverDeno = 0;
    else
        inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);

    scalar_t u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
    scalar_t v = (dot00 * dot12 - dot01 * dot02) * inverDeno;

    w[0] = 1 - u - v;
    w[1] = v;
    w[2] = u;
}

template <typename scalar_t>
std::vector<at::Tensor> forward_rasterize_cpu_kernel(
    at::Tensor face_vertices,
    at::Tensor depth_buffer,
    at::Tensor triangle_buffer,
    at::Tensor baryw_buffer,
    int h,
    int w) {
    
    const int batch_size = face_vertices.size(0);
    const int ntri = face_vertices.size(1);
    
    for (int bn = 0; bn < batch_size; ++bn) {
        for (int i = 0; i < ntri; ++i) {
            const scalar_t* face = face_vertices.data_ptr<scalar_t>() + (bn * ntri + i) * 9;
            scalar_t bw[3];
            point<scalar_t> p0, p1, p2, p;

            p0.x = face[0]; p0.y = face[1];
            p1.x = face[3]; p1.y = face[4];
            p2.x = face[6]; p2.y = face[7];

            int x_min = std::max((int)std::ceil(std::min(p0.x, std::min(p1.x, p2.x))), 0);
            int x_max = std::min((int)std::floor(std::max(p0.x, std::max(p1.x, p2.x))), w - 1);
            int y_min = std::max((int)std::ceil(std::min(p0.y, std::min(p1.y, p2.y))), 0);
            int y_max = std::min((int)std::floor(std::max(p0.y, std::max(p1.y, p2.y))), h - 1);

            for (int y = y_min; y <= y_max; ++y) {
                for (int x = x_min; x <= x_max; ++x) {
                    p.x = x; p.y = y;
                    barycentric_weight(bw, p, p0, p1, p2);

                    if ((bw[2] >= 0) && (bw[1] >= 0) && (bw[0] >= 0)) {
                        scalar_t zp = 1.0 / (bw[0] / face[2] + bw[1] / face[5] + bw[2] / face[8]);
                        
                        // Check if new depth is smaller, and update if it is.
                        if (zp < depth_buffer.data_ptr<scalar_t>()[bn*h*w + y*w + x]) {
                            depth_buffer.data_ptr<scalar_t>()[bn*h*w + y*w + x] = zp;
                            triangle_buffer.data_ptr<int>()[bn*h*w + y*w + x] = i;
                            for (int k = 0; k < 3; ++k) {
                                baryw_buffer.data_ptr<scalar_t>()[bn*h*w*3 + y*w*3 + x*3 + k] = bw[k];
                            }
                        }
                    }
                }
            }
        }
    }
    return {depth_buffer, triangle_buffer, baryw_buffer};
}

template <typename scalar_t>
std::vector<at::Tensor> forward_rasterize_colors_cpu_kernel(
    at::Tensor face_vertices,
    at::Tensor face_colors,
    at::Tensor depth_buffer,
    at::Tensor triangle_buffer,
    at::Tensor images,
    int h,
    int w) {

    const int batch_size = face_vertices.size(0);
    const int ntri = face_vertices.size(1);

    for (int bn = 0; bn < batch_size; ++bn) {
        for (int i = 0; i < ntri; ++i) {
            const scalar_t* face = face_vertices.data_ptr<scalar_t>() + (bn * ntri + i) * 9;
            const scalar_t* color = face_colors.data_ptr<scalar_t>() + (bn * ntri + i) * 9;
            scalar_t bw[3];
            point<scalar_t> p0, p1, p2, p;

            p0.x = face[0]; p0.y = face[1];
            p1.x = face[3]; p1.y = face[4];
            p2.x = face[6]; p2.y = face[7];

            int x_min = std::max((int)std::ceil(std::min(p0.x, std::min(p1.x, p2.x))), 0);
            int x_max = std::min((int)std::floor(std::max(p0.x, std::max(p1.x, p2.x))), w - 1);
            int y_min = std::max((int)std::ceil(std::min(p0.y, std::min(p1.y, p2.y))), 0);
            int y_max = std::min((int)std::floor(std::max(p0.y, std::max(p1.y, p2.y))), h - 1);

            scalar_t cl[3][3];
            for (int num = 0; num < 3; num++) {
                for (int dim = 0; dim < 3; dim++) {
                    cl[num][dim] = color[3 * num + dim];
                }
            }

            for (int y = y_min; y <= y_max; ++y) {
                for (int x = x_min; x <= x_max; ++x) {
                    p.x = x; p.y = y;
                    barycentric_weight(bw, p, p0, p1, p2);

                    if ((bw[2] >= 0) && (bw[1] >= 0) && (bw[0] >= 0)) {
                        scalar_t zp = 1.0 / (bw[0] / face[2] + bw[1] / face[5] + bw[2] / face[8]);
                        
                        if (zp < depth_buffer.data_ptr<scalar_t>()[bn*h*w + y*w + x]) {
                            depth_buffer.data_ptr<scalar_t>()[bn*h*w + y*w + x] = zp;
                            triangle_buffer.data_ptr<int>()[bn*h*w + y*w + x] = i;

                            for (int k = 0; k < 3; ++k) {
                                images.data_ptr<scalar_t>()[bn*h*w*3 + y*w*3 + x*3 + k] = bw[0]*cl[0][k] + bw[1]*cl[1][k] + bw[2]*cl[2][k];
                            }
                        }
                    }
                }
            }
        }
    }
    return {depth_buffer, triangle_buffer, images};
}

std::vector<at::Tensor> forward_rasterize_cpu(
    at::Tensor face_vertices,
    at::Tensor depth_buffer,
    at::Tensor triangle_buffer,
    at::Tensor baryw_buffer,
    int h,
    int w) {
    
    // Dispatch to the correct floating point type
    AT_DISPATCH_FLOATING_TYPES(face_vertices.type(), "forward_rasterize_cpu", ([&] {
        forward_rasterize_cpu_kernel<scalar_t>(
            face_vertices,
            depth_buffer,
            triangle_buffer,
            baryw_buffer,
            h,
            w);
    }));

    return {depth_buffer, triangle_buffer, baryw_buffer};
}


std::vector<at::Tensor> forward_rasterize_colors_cpu(
    at::Tensor face_vertices,
    at::Tensor face_colors,
    at::Tensor depth_buffer,
    at::Tensor triangle_buffer,
    at::Tensor images,
    int h,
    int w) {

    AT_DISPATCH_FLOATING_TYPES(face_vertices.type(), "forward_rasterize_colors_cpu", ([&] {
        forward_rasterize_colors_cpu_kernel<scalar_t>(
            face_vertices,
            face_colors,
            depth_buffer,
            triangle_buffer,
            images,
            h,
            w);
    }));

    return {depth_buffer, triangle_buffer, images};
}
 