#!/bin/bash
git submodule add https://github.com/KindXiaoming/pykan.git
git submodule add https://github.com/facebookresearch/segment-anything.git
git submodule add https://github.com/facebookresearch/sam2.git
git submodule add https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture.git
quarto render Report/Fractography_report.qmd