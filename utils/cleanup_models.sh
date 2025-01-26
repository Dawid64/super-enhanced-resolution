#!/bin/bash

directory="models_copy/"

for file in "$directory"*; do
    if [[ ! "$file" == *"36videos"* || ! "$file" == *"final"* ]]; then
        rm "$file"
    fi
done

