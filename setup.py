from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='FaceIt',  # Name of the package
    version='1.0.0',  # Version of your package
    packages=find_packages(),  # Automatically find all packages in the directory
    url='https://github.com/faezehrabbani97/FaceIt',  # URL to your GitHub repository
    license='MIT',  # License type, can be changed if you use a different license
    author='Faezeh Rabbani',  # Your name
    author_email='faezeh.rabbani97@gmail.com',  # Your email
    description='A pipeline for detecting and analyzing facial movements like eye-tracking and mouse muzzle detection.',  # Short description
    install_requires=requirements,  # Read dependencies from the requirements.txt
    include_package_data=True,
)
