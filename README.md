# Pyxeled
My implementation and adaptation of the paper [*Pixelated Image Abstraction*](https://dl.acm.org/doi/10.5555/2330147.2330154) by Gerstner et al.

This program applies the aesthetic of pixel art to normal photographs. To learn more about how the algorithm works and to see example images it's produced, see [the page dedicated to this project](https://grantskaggs.com/pyxeled) on my website.

### Configuration
This program takes 4 parameters as arguemnts from *stdin:*
* Path to the input image file
* Path to the output image file
* The color palette size (i.e. the number of distinct colors to be present in the output image)
* Width and height dimensions of output image in number of pixels

Each of these four parameters are entered on their own line. For an example of correct format, refer to *config/mountains.txt*.

Additionally, there are a number of configuration variables in *pyxeled.py* which may be useful for development purposes. Most of these variables are artifacts of the Mass Constrained Determenistic Annealing (MCDA) algorithm explained in depth in the above paper. All of these variables are set to values which have emperically performed well on the images I've used in testing this project.

### Execution
There are two recommended ways of executing this project:
1. File redirect for an individual configuration file. For example, `python3 pyxeled.py < config/mountains.txt`
1. Alternatively, the script *run.sh* automates this proces by running *pyxeled.py* on every configuration file in the *config* directory.
