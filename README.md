<div style="border: 2px solid #ccc; padding: 20px; border-radius: 10px;">

  <table style="width: 100%;">
    <tr>
      <td>
        <h1>FACEIT</h1>
        <p>
          <strong>Facial Movement Detection and Analysis Pipeline</strong>
        </p>
        <p>
          <strong>FACEIT</strong> is a comprehensive pipeline for detecting and analyzing facial movements in head-fixed mice, including eye-tracking and muzzle movements. It leverages advanced image processing techniques to capture, track, and analyze facial dynamics, providing valuable insights for various research and experimental applications.
        </p>
      </td>
      <td>
        <img src="figures/Logo_FaceIT.jpg" alt="FACEIT Logo" width="450" style="margin-left: 30px;"/>
      </td>
    </tr>
  </table>

  <p>
    📖 <strong>Explore the full <a href="https://faceit.readthedocs.io/">Documentation</a></strong> for detailed instructions, usage examples, and insights into the pipeline.
  </p>

</div>


## Features

- **Eye-Tracking Analysis**: Capture and analyze eye movements with precision.
- **Mouse Muzzle Tracking**: Detect and monitor muzzle dynamics.
- **Flexible and Modular**: Designed for easy integration and customization.
- **User-Friendly**: Intuitive and interactive GUI for streamlined user interaction and visualization.
- **Multi-Input Support**: Accepts various input formats, including NumPy arrays and video files, ensuring compatibility with diverse workflows.
- **Pupil Analysis Enhancements**: Offers advanced features like blinking detection and saccades analysis to enrich pupil-tracking studies.
- **High-Speed Performance**: Optimized for fast data processing, enabling efficient analysis.
---

# FaceIt - Installation Guide

This guide explains how to install **FaceIt**, ensuring a clean setup with **Python 3.9** 

## 📌 Prerequisites

Before proceeding with the installation, ensure that **Python 3.9** is installed on your system. If it's not available, download and install **Python 3.9** separately from your system's default Python using the link below:  

🔗 [Download Python 3.9](https://www.python.org/downloads/release/python-390/)  

### ⚠️ Important Notes:
- **Do not check** the option **"Add python.exe to PATH"** during installation to avoid conflicts with your system's default Python.
- If you prefer an isolated setup, select **"Customize Installation"** and **specify a dedicated directory** for Python 3.9.
- **Remember the installation path**, as you'll need it later to configure the environment.

---

### 🔹 Step 1: Clone the Repository
We recomend you to use git to clone into your local computer. If you have git open CMD terminal and run:
```bash
git clone https://github.com/faezehrabbani/FaceIt.git
```
If Git is not installed on your system, you can download and install it from the link below:
🔗 [Download Git](https://git-scm.com/downloads)

### 🔹 Alternative: Download the Repository Manually

If you don’t have **Git** installed, you can manually download the repository by following these steps:

1. In the FaceIt Github click on the **"Code"** button (green button).

2. Select **"Download ZIP"**.

3. Extract the ZIP file to your preferred location.

4. Inside the extracted directory, you will find a folder named "FaceIt-main".
Open it and rename the "FaceIt-main" folder inside it to "FaceIt".
Move the FaceIt folder outside of FaceIt-main.
Now, your directory structure should look like this:

```bash
C:\your_preferred_directory\FaceItt
```

### 🔹 Step 2: Navigate to the Project Directory
```bash
cd FaceIt
```


### 🔹 Step 3: Create a Virtual Environment (Using Python 3.9)


If Python 3.9 is not your system's default version, you need to specify its full installation path to create the virtual environment. Use the following command, replacing "C:\path\to\python3.9\python.exe" with the actual location of your Python 3.9 installation:

```bash
"C:\your path to python 3.9\Python39\python.exe" -m venv FaceIt

```
### 🔹 Step 4: Activate the Virtual Environment
Activate the environment to install and run the pipeline without conflicts:

```bash
FaceIt\Scripts\activate
```

### 🔹 Step 5: Install FACEIT
With the virtual environment activated, install the package:

```bash
pip install .
```

Once installed, you can start the application by running:

```bash
faceit
```
If the faceit command is not recognized, try:

```bash
python -m FACEIT_codes.main
```




## 🔄 FaceIt: Daily Workflow

Follow these steps whenever you want to use FaceIt for analysis.

### 🔹 Step 1: Navigate to the FaceIt Directory
Before running the application, open the terminal (CMD or PowerShell) and move to the FaceIt project folder:

```bash
cd FaceIt
```
### 🔹 Step 2: Activate the Virtual Environment
Since FaceIt runs inside a virtual environment, it must be activated before use.

```bash
FaceIT\Scripts\activate
```
### 🔹 Step 3: Run FaceIt
Once the environment is active, start the application by typing:

```bash
faceit
```

experimental procedures followed French and European guidelines for animal experimentation and in compliance with the institutional animal welfare guidelines of the Paris Brain Institute

## Contributing

Contributions to FACEIT are welcome! Feel free to:

- Report issues.
- Submit pull requests.
- Suggest new features.

---

## License

FACEIT is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## Support

If you encounter issues or have questions, please contact **Faezeh Rabbani** at:
📧 **[faezeh.rabbani97@gmail.com](mailto:faezeh.rabbani97@gmail.com)**

## Acknowledgments

This pipeline was developed in the **Bacci Lab** at the **Paris Brain Institute**.
For more information about the Bacci Lab, visit [the official website](https://baccilab.org).
