from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qdseg",
    version="0.2.1",
    description="Quantum Dot Segmentation and Analysis Tool for AFM/XQD files",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="jkkwoen",
    url="https://github.com/jkkwoen/qdseg",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.20.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "stardist": [
            "stardist>=0.8.0",
            "tensorflow>=2.10.0",
        ],
        "cellpose": [
            "cellpose>=3.0.0",
        ],
        "training": [
            "torch>=2.0.0",
            "zarr>=2.16.0",
            "tqdm>=4.65.0",
        ],
        "all": [
            "stardist>=0.8.0",
            "tensorflow>=2.10.0",
            "cellpose>=3.0.0",
            "torch>=2.0.0",
            "zarr>=2.16.0",
            "tqdm>=4.65.0",
        ],
    },
    python_requires=">=3.8",
    keywords="quantum-dot qd afm segmentation nanoparticle image-processing",
)
