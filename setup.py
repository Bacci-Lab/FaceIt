from setuptools import setup, find_packages

# Read the README file for a detailed package description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="faceit",
    version="1.0.0",
    description="FACEIT is a pipeline to analyze facial movements, such as eye movements and the mouse's muzzle.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Faezeh Rabbani",
    author_email="faezeh.rabbani97@gmail.com",
    packages=find_packages(include=["FACEIT_codes", "FACEIT_codes.*"]),
    install_requires=[
        "attrs>=23.2.0",
        "colorama>=0.4.6",
        "opencv-python>=4.9.0.80",
        "pandas>=2.0.0",
        "PyQt5>=5.15.9",
        "scikit-learn==1.2.2",
        "matplotlib>=3.8.4",
        "numpy>=1.24.4",
        "scipy>=1.9.1",
        "bottleneck>=1.3.7,<2.0.0",
        "pynwb>=2.0.0",
        "tqdm>=4.0.0"
    ],
    extras_require={
        "gui": ["PyQt5", "pyqtgraph"],
        "io": ["h5py", "pynwb"]
    },
    entry_points={
        "console_scripts": [
            "faceit=FACEIT_codes.main:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
)
