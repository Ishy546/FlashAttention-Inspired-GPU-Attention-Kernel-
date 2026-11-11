from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
 
setup(
    name="flashattn_lite",
    ext_modules=[
        CUDAExtension(
            name="flashattn_lite",
            sources=["csrc/attention.cpp", "csrc/attention_kernels.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "--use_fast_math", "-lineinfo"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
 
