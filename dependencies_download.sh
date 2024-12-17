#!/bin/bash
git submodule add https://github.com/KindXiaoming/pykan.git
git submodule add https://github.com/facebookresearch/segment-anything.git
git submodule add https://github.com/facebookresearch/sam2.git
git submodule add https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture.git

# Set the URL of the file to download
url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Set the name of the file to save
filename="sam_vit_h_4b8939.pth"

# Create the directory to save the file (if it doesn't already exist)
mkdir -p "$(dirname "$filename")"

# Download the file using curl
curl -L "$url" -o "$filename"

echo "File downloaded: $filename"

quarto render Report/Fractography_report.qmd