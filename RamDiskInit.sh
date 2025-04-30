#!/bin/bash

ramdisk=/mnt/ramdisk
sudo mount -t tmpfs -o rw,size=8G tmpfs $ramdisk
cp -r ./data $ramdisk
