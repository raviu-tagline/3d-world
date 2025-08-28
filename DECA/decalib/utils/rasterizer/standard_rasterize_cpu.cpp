#include <torch/torch.h>
#include <vector>
#include <iostream>

// Forward declarations for the CPU-only functions
std::vector<at::Tensor> forward_rasterize_cpu(
        at::Tensor face_vertices,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor baryw_buffer,
        int h,
        int w);

std::vector<at::Tensor> standard_rasterize(
        at::Tensor face_vertices,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor baryw_buffer,
        int height, int width
        ) {
    // Call the CPU-specific function
    return forward_rasterize_cpu(face_vertices, depth_buffer, triangle_buffer, baryw_buffer, height, width);
}

// Forward declarations for the CPU-only color functions
std::vector<at::Tensor> forward_rasterize_colors_cpu(
        at::Tensor face_vertices,
        at::Tensor face_colors,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor images,
        int h,
        int w);

std::vector<at::Tensor> standard_rasterize_colors(
        at::Tensor face_vertices,
        at::Tensor face_colors,
        at::Tensor depth_buffer,
        at::Tensor triangle_buffer,
        at::Tensor images,
        int height, int width
        ) {
    // Call the CPU-specific color function
    return forward_rasterize_colors_cpu(face_vertices, face_colors, depth_buffer, triangle_buffer, images, height, width);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Expose the functions with the correct names
    m.def("rasterize", &standard_rasterize, "RASTERIZE (CPU)");
    m.def("rasterize_colors", &standard_rasterize_colors, "RASTERIZE COLORS (CPU)");
}
