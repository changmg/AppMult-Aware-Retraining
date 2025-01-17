# This setup.py refers to the open-source project: https://github.com/YuxueYang1204/CudaDemo

import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


source_folder = './self_ops/src/'
project_path = os.path.dirname(os.path.abspath(__file__))


debug_mode = False
extra_link_args = [
    # "-lblas",
]
extra_compile_args = {
    "cxx": [
        "-O3" if not debug_mode else "-O0"
    ],
    "nvcc": [
        "-O3" if not debug_mode else "-O0"
    ],
}
if debug_mode:
    extra_compile_args["cxx"].append("-g")
    extra_compile_args["nvcc"].append("-g")
    extra_link_args.extend(["-O0", "-g"])


setup(
    name='approx_ops',
    packages=find_packages(),
    version='0.1.0',
    author='Chang Meng',
    ext_modules=[
        CUDAExtension(
            'approx_ops',
            [f'{source_folder}/op_register.cpp',
             f'{source_folder}/approx_gemm_forward.cu',
             f'{source_folder}/approx_gemm_backward.cu',
            #  f'{source_folder}/approx_bmm_forward.cu',
            #  f'{source_folder}/approx_bmm_backward.cu',
            #  f'{source_folder}/approx_conv2d_forward.cu',
            #  f'{source_folder}/approx_conv2d_backward.cu',
             ],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)