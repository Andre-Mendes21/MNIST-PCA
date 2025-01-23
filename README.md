# Digit Recognition Using PCA With SVD

This small project uses the PCA algorithm with both euclidean and Mahalanobis distances to classify and recognize digits in the [MNIST Database](http://yann.lecun.com/exdb/mnist/).

## Setup
Run:
```console
$ ./setup.sh
```
Uncompress the files in the `/data/test` and `/data/training` then run from the project root:
```console
$ python -m venv .env
$ source .env/Scripts/activate
$ pip install numpy matplotlib scikit-images
```

## Usage
```console
$ python main.py
```

## References
- http://yann.lecun.com/exdb/mnist/
- http://colah.github.io/posts/2014-10-Visualizing-MNIST/
- http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf
