# Research Paper

## The Use of Machine Learning in Predicting Neurological Disorders for Epilepsy

- **Author**: Saniya Vaish, John P. Stevens High School, Edison, NJ, USA ([3017061@gmail.com](mailto:3017061@gmail.com))
- **Journal**: International Journal for Multidisciplinary Research (IJFMR)
- **Volume/Issue**: 6(3), May-June 2024
- **ISSN**: 2582-2160
- **DOI**: [10.36948/ijfmr.2024.v06i03.22870](https://doi.org/10.36948/ijfmr.2024.v06i03.22870)
- **Full text**: [The Use of Machine Learning in Predicting Neurological Disorders for Epilepsy.pdf](The%20Use%20of%20Machine%20Learning%20in%20Predicting%20Neurological%20Disorders%20for%20Epilepsy.pdf)

Author contributions: concept design, methodology, data analysis, implementation, and manuscript preparation.

### Abstract

Epilepsy affects over 50 million people worldwide, with diagnosis often relying on EEG signals and clinical expertise. This research investigates machine learning approaches for automated epilepsy prediction and neurological disorder detection. We implement and compare multiple classical machine learning models (Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes) and deep neural networks on biomedical datasets. Our methodology includes comprehensive data preprocessing, feature engineering with polynomial and ratio-based features, and rigorous cross-validation evaluation. Results demonstrate that ensemble methods (Random Forest, Gradient Boosting) achieve superior performance for epilepsy classification. The implementation provides reproducible workflows suitable for clinical decision support systems. This work contributes to the field of computational neurology by providing scalable, interpretable ML pipelines for neurological disorder prediction.

**Keywords**: epilepsy detection, EEG signal processing, machine learning classification, neurological disorder prediction, computational neurology, deep learning for healthcare, feature engineering, biomedical signal analysis.

## Methods → code

| Method | Implementation |
|--------|----------------|
| Missing-value imputation (median), standardization, outlier/artifact handling | [src/preprocessing/pipeline.py](../src/preprocessing/pipeline.py), [src/preprocessing/eeg_filters.py](../src/preprocessing/eeg_filters.py) |
| Ratio and statistical features, degree-2 polynomial features, f-statistic feature selection | [src/features/engineer.py](../src/features/engineer.py), [src/preprocessing/pipeline.py](../src/preprocessing/pipeline.py) |
| EEG signal features (time, frequency, wavelet, Hjorth) | [src/features/extractor.py](../src/features/extractor.py) |
| Logistic regression, decision tree, random forest (200 trees), gradient boosting, SVM (RBF), KNN, naive Bayes | [src/models/classical.py](../src/models/classical.py) |
| PyTorch MLP with dropout | [src/models/deep.py](../src/models/deep.py) |
| Stratified cross-validation, GridSearchCV | [src/training/train.py](../src/training/train.py) |
| Accuracy, precision, recall, F1, ROC-AUC, sensitivity, specificity | [src/evaluation/metrics.py](../src/evaluation/metrics.py) |
| EDA, correlation heatmaps, PCA, confusion matrices, feature importance | [src/visualization/plots.py](../src/visualization/plots.py) |

Datasets: UCI Epileptic Seizure Recognition (when available locally or via OpenML) and the UCI
Breast Cancer Wisconsin and Wine datasets bundled with scikit-learn. Sources and citations are
listed in [src/data/loader.py](../src/data/loader.py).

### Reproducibility

Results can be reproduced with the notebooks at the repository root; see the
[main README](../README.md) for setup. Fixed random seeds (`random_state=42`,
[src/utils/seed.py](../src/utils/seed.py)) keep runs repeatable.

## Citation

**BibTeX**:

```bibtex
@article{vaish2024epilepsy,
  author = {Vaish, Saniya},
  title = {The Use of Machine Learning in Predicting Neurological Disorders for Epilepsy},
  journal = {International Journal for Multidisciplinary Research (IJFMR)},
  volume = {6},
  number = {3},
  pages = {May-June},
  year = {2024},
  issn = {2582-2160},
  doi = {10.36948/ijfmr.2024.v06i03.22870},
  url = {https://www.ijfmr.com}
}

@software{vaish2026showcaserepo,
  author = {Vaish, Saniya},
  title = {Biomedical ML Research Showcase: Epilepsy Prediction Pipeline},
  year = {2026},
  url = {https://github.com/3017061/epilepsy-ml-pipeline},
  note = {Open-source implementation accompanying IJFMR publication}
}
```

**APA**:

Vaish, S. (2024). The use of machine learning in predicting neurological disorders for epilepsy. *International Journal for Multidisciplinary Research (IJFMR)*, 6(3), May-June. <https://doi.org/10.36948/ijfmr.2024.v06i03.22870>

Vaish, S. (2026). Biomedical ML research showcase: Epilepsy prediction pipeline [Computer software]. <https://github.com/3017061/epilepsy-ml-pipeline>
