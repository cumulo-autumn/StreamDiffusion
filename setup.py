import re

from setuptools import find_packages, setup


_deps = [
    "torch",
    "xformers",
    "diffusers",
    "transformers",
    "accelerate",
]

deps = {
    b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)
}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}
extras["xformers"] = deps_list("xformers")
extras["torch"] = deps_list("torch", "accelerate")

extras["dev"] = extras["xformers"] + extras["torch"]

install_requires = [
    deps["diffusers"],
    deps["transformers"],
]

setup(
    name="streamlcm",
    version="0.1.0",
    description="",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="",
    license="Apache",
    author="",
    author_email="",
    url="https://github.com/cumulo-autumn/StreamLCM",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"streamlcm": ["py.typed"]},
    include_package_data=True,
    python_requires=">=3.10.0",
    install_requires=list(install_requires),
    extras_require=extras,
)
