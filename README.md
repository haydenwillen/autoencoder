# Autoencoder vs. PCA on MNIST

**Hayden Willen** | Machine Learning II | University of Virginia School of Data Science

This project builds an autoencoder in PyTorch that compresses 28x28 MNIST digits into a 2-D bottleneck, reconstructs them, and compares that 2-D representation with a 2-D PCA projection. It follows Michelucci (2022), [An Introduction to Autoencoders](https://arxiv.org/abs/2201.03898).

Compressing data into a small number of features is used for visualization, anomaly detection, denoising, and faster downstream models, so it matters which method keeps the most useful structure.

## What's in the notebook

1. **Background.** Short answers on what autoencoders do, the role of the bottleneck, their applications, and how they differ from PCA.
2. **Data preparation.** Each image is scaled to [0,1] using its own min and max, then split 70/15/15 into fitting, tuning, and reporting sets.
3. **Model and training.** A 784 → 128 → 64 → 2 → 64 → 128 → 784 autoencoder with BatchNorm, ReLU, and Dropout, a sigmoid output, binary cross-entropy loss, and early stopping.
4. **Reconstruction.** Original and reconstructed digits side by side from the reporting set.
5. **Learned 2-D representation.** A scatterplot of the bottleneck features colored by digit.
6. **Comparison with PCA.** A 2-component PCA fit on the fitting and tuning sets, plotted the same way.

## Results

| Method | 2-D representation | Result on reporting set |
|---|---|---|
| Autoencoder | 2-unit bottleneck, non-linear | BCE **0.1897** |
| PCA | 2 principal components, linear | **16.89%** of variance explained |

Training stopped early at epoch 33 and kept the best weights from epoch 28 (tuning BCE 0.1899).

## Key findings

- **The model did not overfit.** The reported BCE of 0.1897 is almost the same as the best tuning BCE of 0.1899.
- **Reconstructions are recognizable but blurry.** Every image passes through only 2 numbers, so fine detail is lost.
- **Similar digits get mixed up.** 4 and 9, and 3 and 8, sometimes reconstruct as each other, while digits like 2 and 6 come back consistently.
- **The autoencoder separates digits much better than PCA.** Most digits get their own region in the autoencoder's 2-D space, while in PCA only the digit 1 forms a clear group.
- **Non-linear compression keeps more structure.** Two linear PCA components capture only about 17% of the variance, which is why the PCA plot overlaps so much.

## Data

The notebook uses the [MNIST dataset](https://www.openml.org/d/554) (LeCun, Cortes and Burges), 70,000 handwritten digit images. It downloads automatically from OpenML with `fetch_openml("mnist_784")`, so no data file is stored in this repo.

## How to run

```bash
git clone https://github.com/haydenwillen/autoencoder.git
cd autoencoder
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook autoencoder.ipynb
```

## Project structure

```
autoencoder/
├── autoencoder.ipynb   # full analysis with outputs
├── README.md           # this file
├── requirements.txt    # Python packages
└── .gitignore          # files Git ignores
```

## Limitations

- I chose a 2-D bottleneck so I can plot the features. A larger bottleneck would give sharper reconstructions.
- The reconstruction figure shows only the first 10 images of the reporting set.
- The random seed is set once at the top, so exact numbers can shift slightly if cells are run out of order. Use Run All for the same results.
- The comparison with PCA is visual. There is no clustering score or classifier trained on the 2-D features.
- The first run needs an internet connection to download MNIST.

## Credits

This started as a group lab for Machine Learning II at the University of Virginia. I wrote all of the code and answers in this notebook. I used Claude to help clean it up.
