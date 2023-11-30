import re

from setuptools import find_packages, setup


_deps = [
    "torch",
    "xformers",
    "diffusers",
    "transformers",
    "accelerate",
    "fire",
    "omegaconf",
    "pywin32",
    "cuda-python",
    "onnx==1.13.1",
    "onnxruntime==1.14.1",
    "colored"
]

deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}
extras["xformers"] = deps_list("xformers")
extras["torch"] = deps_list("torch", "accelerate")
extras["tensorrt"] = deps_list("pywin32", "cuda-python", "onnx", "onnxruntime", "colored")

extras["dev"] = extras["xformers"] + extras["torch"] + extras["tensorrt"]

install_requires = [
    deps["fire"],
    deps["omegaconf"],
    deps["diffusers"],
    deps["transformers"],
]

setup(
    name="stream-diffusion",
    version="0.1.0",
    description="",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="",
    license="Apache",
    author="",
    author_email="",
    url="https://github.com/cumulo-autumn/StreamDiffusion",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"streamdiffusion": ["py.typed"]},
    include_package_data=True,
    python_requires=">=3.10.0",
    install_requires=list(install_requires),
    extras_require=extras,
)
