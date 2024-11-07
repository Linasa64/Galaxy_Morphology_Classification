#!/bin/bash

source ../.venv/bin/activate

packages=("numpy" "matplotlib" "scipy" "astropy" "scikit-image" "opencv-python-headless" "tensorflow" "torch" "torchvision" "pillow" "tqdm" "pandas")

failed_packages=()

for package in "${packages[@]}"; do
    echo "Installation de $package..."
    if ! pip install "$package"; then
        echo "Erreur lors de l'installation de $package"
        failed_packages+=("$package")  # Ajouter à la liste des échecs
    fi

    echo "Nettoyage du cache pip..."
    pip cache purge
done

if [ ${#failed_packages[@]} -gt 0 ]; then
    echo "Les packages suivants n'ont pas pu être installés :"
    for pkg in "${failed_packages[@]}"; do
        echo "- $pkg"
    done
else
    echo "Tous les packages ont été installés avec succès."
fi
