#!/bin/bash

# Loop through each dataset folder in the directory
for folder in *_qna
do
    cd "$folder"
    tfds build --overwrite
    cd ..
done