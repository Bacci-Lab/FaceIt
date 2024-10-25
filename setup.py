from setuptools import setup, find_packages

# Read the requirements from the requirements.txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='FaceIt',
    version='1.0.0',
    packages=find_packages()
    package_data=['FACEIT_codes'],  # Explicitly list the FaceIt package
    url='https://github.com/faezehrabbani97/FaceIt',
    license='MIT',
    author='Faezeh Rabbani',
    author_email='faezeh.rabbani97@gmail.com',
    description='A pipeline for detecting and analyzing facial movements like eye-tracking and mouse muzzle detection.',
    install_requires=requirements,
    include_package_data=True,  # Ensure additional files are included
    python_requires='>=3.6',
)
