from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='FaceIt',
    version='1.0.0',
    packages=find_packages(),  # Automatically finds packages
    package_data={'FaceIt': ['FACEIT_codes/*']},  # Ensure FACEIT_codes is included
    url='https://github.com/faezehrabbani97/FaceIt',
    license='MIT',
    author='Faezeh Rabbani',
    author_email='faezeh.rabbani97@gmail.com',
    description='A pipeline for detecting and analyzing facial movements.',
    install_requires=requirements,
    
)
