# Copyright (c) Zhihao Liang. All rights reserved.
import os
from typing import List
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_requirements(filename: str="requirements.txt") -> List[str]:
    assert os.path.exists(filename), f"{filename} not exists"
    with open(filename, "r") as f:
        content = f.read()
    lines = content.split("\n")
    requirements_list = list(
        filter(lambda x: x != "" and not x.startswith("#"), lines))
    return requirements_list


def get_version() -> str:
    version_file = os.path.join("custma", "version.py")
    with open(version_file, "r", encoding="utf-8") as f:
        exec(compile(f.read(), version_file, "exec"))
    return locals()["__version__"]


def get_extensions():
    ext_modules = [
        CUDAExtension(
            name="custma.src",
            sources=[
                "custma/src/bindings.cpp",
                "custma/src/stereo_matching.cpp",
                "custma/src/stereo_matching_kernel.cu",
            ],
            include_dirs=[os.path.join(ROOT_DIR, "custma", "include")],
            optional=False),
    ]
    return ext_modules


setup(
    name="custma",
    version=get_version(),
    author="Zhihao Liang",
    author_email="eezhihaoliang@mail.scut.edu.cn",
    description="CUDA implementation of marching cubes",
    url="https://github.com/lzhnb/CuStereoMatching",
    # long_description=open("README.md").read(),
    license="BSD-3",
    ext_modules=get_extensions(),
    setup_requires=["pybind11>=2.5.0"],
    packages=["custma", "custma.src"],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
)
