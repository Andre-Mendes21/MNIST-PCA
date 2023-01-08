# Digit Recognition Using PCA

This small project uses the PCA algorithmn with both euclidean and Mahalanobis distances to classify and recognize digits in the [MNIST Database](http://yann.lecun.com/exdb/mnist/).

## Setup
```console
python -m venv .env
source .env/Scripts/activate
pip install numpy matplotlib
```

## Usage
```console
python main.py
```
## TODO
- [x] Reading data from MNIST Dataset
- [x] Define the principal components of the data
- [x] Implement euclidean and Mahalanobis distances
- [x] Test the model with MNIST test data
- [ ] Test model with non-MNIST data (e.g. letters, noise, etc.)


## References
- http://colah.github.io/posts/2014-10-Visualizing-MNIST/
- http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf