### Step 1: Install all dependencies
**Recommended:** Create a virtual environment. If you don't know what they are, please follow this [tutorial](https://www.w3schools.com/python/python_virtualenv.asp). It is very helpful and used all the time in python, so it's a skill you need to learn.

```pip install -r requirements.txt```
**Note:** If you add any new dependencies for your PR, add it to requirements.txt

## Perspective Correction

Everything in this directory is a modified version of [this article](https://medium.com/@siromermer/extracting-chess-square-coordinates-dynamically-with-opencv-image-processing-methods-76b933f0f64e) to work for our chess board.  

```perspective_correction.py``` takes an input image, does transformations on it to remove all perspective, detects the edges, and removes the transformations with the overlay of the edges.  

You can follow ```perspective_correction.ipynb``` for the step by step process. It is heavily commented and documented.  
If you've never worked with a jupyter notebook before, you can basically run python code in blocks instead of all at once. It is very helpful for this kind of iterative step by step process.  

All test images are in the ```test_images``` directory.