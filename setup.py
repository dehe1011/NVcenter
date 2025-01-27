from setuptools import find_packages, setup

with open("requirements/requirements.txt") as f:
    required = f.read().splitlines()

with open("requirements/dev-requirements.txt") as f:
    dev = f.read().splitlines()

extras = dev


def read_version():
    with open("NVcenter/__init__.py") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"')

setup(
    name="NVcenter",
    version=read_version(),
    author="Dennis Herb",
    author_email="dennis.herb@uni-ulm.de",
    description="A package to perform simulations on NV centers in spin baths.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dehe1011/NVcenter",
    license="BSD-3-Clause",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    extras_require={"dev": dev, "all": extras},
)