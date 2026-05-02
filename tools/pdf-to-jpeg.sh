#!/bin/bash

if [ $1==pcd ]; then
    Dir=~/Desktop/Privitera_335/panaSKImg_output/
    if [ -d "$Dir$2/PCD_jpeg" ]; then
        continue
    else
        mkdir $2/PCD_jpeg
    fi
    for i in {1,2,3,4}; do pdftoppm -jpeg -r 300 -cropbox $Dir$2 EXT$i.pdf $Dir$2/PCD_jpeg/$1$i; done
elif [ $1==plot ]; then
    Dir=./
    pdftoppm -jpeg -r 300 -cropbox $2.pdf $Dir$2
fi