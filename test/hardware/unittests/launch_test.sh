#!/bin/bash

files=$(ls test*.py)

for f in $files; do
    module=$(echo $f | cut -d'.' -f1);
    
done