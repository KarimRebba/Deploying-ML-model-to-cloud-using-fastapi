# Model Card

## Model Details
- Developed by Karim Rebba for the Udacity project "Deploying an ML Model to a Cloud Application Platform with FastAPI"
- Date: October 2026
- Model type: Random Forest classifier
- Hyperparameters: n_estimators=100, random_state=42
- Preprocessing:
    * Leading and trailing white spaces are removed from the values of the dataset
    * Categorial features are one-hot encoded with a fitted OneHotEncoder
    * Target is binarized with a fitted LabelBinarizer
- Artifacts required for inference to avoid retraining:
    * Trained model in model/rfc_model.joblib
    * Fitted OneHotEncoder in model/encoder.joblib
    * Fitted LabelBinarizer in model/label_binarizer.joblib

## Intended Use
- Predicts whether a person's annual income is above or below $50K (>50K / <=50K) based on various attributes from census data
- Intended for educational use only and not for real-world decisions

## Training Data
- Source: the UCI Census Income dataset, extracted from the [1994 US Census database](https://archive.ics.uci.edu/dataset/20/census+income).
- The dataset has 32,561 rows, 14 features and the binary target "salary".
- Cleaning: leading and trailing whitespace were removed from column names and values. The cleaned data was saved in ``data/census_clean.csv``. Columns were not edited. Missing values marked as ? were also left as is.
- Split: 80% of the data was used for training.
- Categorical features: workclass, education, marital-status, occupation, relationship, race, sex, native-country.

## Evaluation Data
- A fifth of the data is held out for testing. The testing data is processed using the encoder and label binarizer fitted on the training data

## Metrics
The model is tested on the high income class (>50K) with precision, recall and F1.

| Metric |	Test set |
|--------|-----------|
| Precision |	0.7622 |
| Recall |	0.6384 |
| F1	| 0.6948 |

The model is fairly precise predicting >50K category, but it misses about 36% of actual high earners.

*Performance on slices:* metrics for every value of each categorical feature are in ``slice_output.txt``.

- Key observations:

    * Sex: recall is lower for women (0.524, F1 0.618) than for men (0.658, F1 0.707).
    * Education: performance is strong for Masters (F1 0.850), Doctorate (F1 0.881) and Prof-school (F1 0.894). It is weak for HS-grad (F1 0.505), 10th (F1 0.267) and 7th-8th (recall/F1 0.0).
    * Small slices: many native-country values have fewer than 15 samples. Their metrics are extreme, usually either 0.0 or 1.0 and *are not statistically meaningful*.

## Ethical Considerations
- The dataset contains sensitive data such as race, sex, native-country. The model can reproduce historical income disparities tied to these attributes.

## Caveats and Recommendations
- "?" Values are treated as their own category. Handling missing values explicitly might enhance the performance
- Many slices are small. Merging rare categories might be a better option for non-educational use
- This model uses a train/test split instead of k-fold cross-validation, which combined with hyperparameter tuning could improve the performance