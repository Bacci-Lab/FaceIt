from setuptools import setup, find_packages

setup(
    name="faceit",
    version="1.0.0",
    description="FACEIT is a pipeline to analyze facial movements, such as eye movements and the mouse's muzzle",
    author="Faezeh Rabbani",
    author_email="faezeh.rabbani97@gmail.com",
    packages=find_packages(),
    install_requires=[
        "attrs>=23.2.0,<24.0.0",
        "colorama>=0.4.6,<0.5.0",
        "opencv-python>=4.9.0.80,<5.0.0",
        "pandas>=2.0.0,<2.1.0",
        "PyQt5>=5.15.9,<5.16.0",
        "scikit-learn>=1.2.2,<1.3.0",
        "matplotlib>=3.8.4,<3.9.0",
        "numpy>=1.24.4,<1.25.0",
        "scipy>=1.9.1,<2.0.0"
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
)
