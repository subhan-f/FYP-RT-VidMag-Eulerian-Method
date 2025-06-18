# Test Specification

Unit tests are located in the `tests/` folder and exercise over 90% of the code
base using small synthetic videos.

## Running
```bash
pytest -q tests/
```

## Coverage
- frame extraction and cropping
- stabilisation and Laplacian reconstruction
- temporal filtering and motion magnification
- feature generation and PCA
- SVM training and full inference

Synthetic frames are generated on the fly so the total test data remains under
2&nbsp;MB.
