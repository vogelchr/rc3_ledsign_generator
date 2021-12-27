#!/bin/sh

for pixel in *_pixels.png ; do
	json="${pixel%_pixels.png}_animation.json"
	./ledsign_generator.py $pixel $json
done
