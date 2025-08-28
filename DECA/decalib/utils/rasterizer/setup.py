# To install, run
# python setup.py build_ext -i
# Ref: https://github.com/pytorch/pytorch/blob/11a40410e755b1fe74efe9eaa635e7ba5712846b/test/cpp_extensions/setup.py#L62

from setuptools import setup
# Import CppExtension to build for CPU
from torch.utils.cpp_extension import BuildExtension, CppExtension
import os

# USE_NINJA = os.getenv('USE_NINJA') == '1'
os.environ["CC"] = "clang"
os.environ["CXX"] = "clang"

USE_NINJA = os.getenv('USE_NINJA') == '1'

setup(
    # Change the extension name to 'standard_rasterize_cpu'
    name='standard_rasterize_cpu',
    ext_modules=[
        # Change CUDAExtension to CppExtension
        CppExtension('standard_rasterize_cpu', [
            # Use the CPU source files
            'standard_rasterize_cpu.cpp',
            'standard_rasterize_cpu_kernel.cc',
        ])
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=USE_NINJA)}
)