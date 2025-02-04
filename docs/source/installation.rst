Instalation
============

This guide explains how to install **FaceIt**, ensuring a clean setup with **Python 3.9**

Prerequisites
-------------

Before installing, make sure you have:

- **Python 3.9** installed separately from your system Python:
  `Download Python 3.9 <https://www.python.org/downloads/release/python-390/>`_
- **Git** installed:
  `Download Git <https://git-scm.com/downloads>`_

---

Step 1: Clone the Repository
----------------------------

Open a **Command Prompt (CMD)** terminal and run:

.. code-block:: bash

    git clone https://github.com/faezehrabbani/FaceIt.git

Step 2: Navigate to the Project Directory
-----------------------------------------

Move into the project folder:

.. code-block:: bash

    cd FaceIt

Step 3: Create a Virtual Environment (Using Python 3.9)
--------------------------------------------------------

If Python 3.9 is not your system's default version, you need to specify its full installation path to create the virtual environment.

replace `"C:\path\to\python3.9\python.exe"` with the actual location of **Python 3.9**:

.. code-block:: bash

    "C:\path\to\python3.9\python.exe" -m venv FaceIt_env

Step 4: Activate the Virtual Environment
----------------------------------------

Activate the environment to install and run the pipeline without conflicts.

**On Windows (CMD):**

.. code-block:: bash

    FaceIt_env\Scripts\activate


Step 5: Install FaceIt
----------------------

With the virtual environment activated, install the package:

.. code-block:: bash

    pip install .

Running FaceIt
--------------

Once installed, you can start the application by running:

.. code-block:: bash

    faceit

If the **faceit** command is not recognized, try:

.. code-block:: bash

    python -m FACEIT_codes.main
