This repository contains training and example code from "Multi-Modal Masked Autoencoders for Learning Image-Spectrum Associations for Galaxy Evolution and Cosmology" by Morgan Himes, Samiksha Krishnamurthy, Andrew Lizarraga, Srinath Saikrishnan, Vikram Seenivasan, Jonathan Soriano, Ying Nian Wu, and Tuan Do.

Python scripts for training and a plotting notebook are included. The requirements.txt file details the packages to be installed. The model class is contained in modules.py. The custom loader for the data is contained in galaxy_loader.py. Visualize results and recreate plots from the paper using MMAE_NeurIPS2025_Plots_Anonymized.ipynb. The file photoz_utils.py contains some functions necessary for producing the redshift regression plot (see https://github.com/astrodatalab/datalabutils).

Training data is the GalaxiesML-Spectra Dataset, available at https://zenodo.org/uploads/16989593

To run the model, replace placeholder data paths with the location of the GalaxiesML-Spectra HDF5 file. The training script is built to use MLflow (https://mlflow.org/) for model logging; run with MLflow or replace MLflow with your preferred logging method. Run "python train.py" (with a number of editable parameters, see file) to start the training.
