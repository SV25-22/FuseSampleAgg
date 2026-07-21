from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent

setup(
    name="fusesampleagg",
    version="1.0.0",
    description="Fused CUDA neighbor sampling and mean aggregation for PyTorch",
    packages=["fuseop"],
    ext_modules=[
        CUDAExtension(
            name="fuseop._C",
            sources=[
                str(ROOT / "fuseop" / "binding.cpp"),
                str(ROOT / "fuseop" / "fused_sample_agg.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    python_requires=">=3.10",
)
