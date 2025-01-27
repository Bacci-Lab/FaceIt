import os

# Open the requirements.txt file
with open("requirements.txt", "r") as file:
    dependencies = file.readlines()

# Loop through each dependency and add it to Poetry
for dependency in dependencies:
    os.system(f"poetry add {dependency.strip()}")
