import os
import re
from setuptools import setup, find_packages

# Read version from __init__.py
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), "src", "concord", "__init__.py")
    with open(version_file, "r") as f:
        match = re.search(r'__version__ = "(.*?)"', f.read())
        if match:
            return match.group(1)
        raise RuntimeError("Version not found in src/concord/__init__.py")

setup(
    name='concord-sc',  # Unique PyPI package name
    version=get_version(),
    description='CONCORD: Contrastive Learning for Cross-domain Reconciliation and Discovery',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),  # Lowercase to match import conventions
    package_dir={"": "src"},
    author='Qin Zhu',
    author_email='qin.zhu@ucsf.edu',
    url='https://github.com/Gartner-Lab/Concord',
    license="MIT",
    install_requires=[
        "anndata>=0.8",
        "numpy>=1.23",
        "h5py>=3.1",
        "tqdm",
        "umap-learn>=0.5.1",
        "matplotlib>=3.6",
        "pandas>=1.5",
        "plotly>=5.0.0",
        "scanpy>=1.1",
        "scikit-learn>=0.24",
        "scipy>=1.8",
        "seaborn>=0.13",
        "scikit-misc",
        "nbformat",
        "build"
    ],
    extras_require={
        "optional": [
            "gseapy>=1.1.0",
            "Pillow>=10.0.0",
            "plottable>=0.1",
            "requests>=2.0",
            "rpy2>=3.5"
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
