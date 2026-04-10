#!/bin/bash
# Download and extract the Sintel dataset components used by the CUT3R eval pipeline
echo "Downloading MPI Sintel Training Images... (1.3GB)"
wget -q --show-progress http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_images.zip

echo "Downloading MPI Sintel Camera Data... (11MB)"
wget -q --show-progress http://files.is.tue.mpg.de/sintel/MPI-Sintel-training_camdata.zip

echo "Extracting archives..."
unzip -q MPI-Sintel-training_images.zip
unzip -q MPI-Sintel-training_camdata.zip

echo "Structuring for CUT3R..."
mkdir -p data/sintel/training
mv training/final data/sintel/training/
mv training/camdata_left data/sintel/training/

echo "Cleaning up..."
rm -rf training
rm MPI-Sintel-training_images.zip MPI-Sintel-training_camdata.zip

echo "Sintel Zero-Shot dataset is ready in data/sintel/"
