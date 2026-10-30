# Deploying a ML model to predict income from census data using FastAPI
The project trains a random forest classifier to predict the predict if a person earns more or less than $50k. For more information about the model see the ``model_card.md``. The model is served using FastAPI and GitHub actions and Render are used for CI/CD.

The GitHub repo is located [here](https://github.com/KarimRebba/Deploying-ML-model-to-cloud-using-fastapi)

## File structure:
.
├── .github/workflows/quality_control.yml   # CI (pytest + flake8) and deployment
├── data/
│   ├── census.csv                          # Raw data
│   └── census_clean.csv                    # Data with whitespace removed
├── model/
│   ├── rfc_model.joblib                    # Trained RandomForestClassifier
│   ├── encoder.joblib                      # Fitted OneHotEncoder
│   └── label_binarizer.joblib              # Fitted LabelBinarizer
├── screenshots/                            # Rubric screenshots
├── starter/
│   ├── ml/
│   │   ├── data.py                         # process_data
│   │   └── model.py                        # train, save/load, inference, metrics, slices
|   ├── slice_output.txt                    # Metrics on categorical slices
│   └── train_model.py                      # Training script
├── tests/                                  # Model and API unit tests
├── clean.ipynb                             # Remove whitespaces / explore data
├── main.py                                 # FastAPI app
├── live_post.py                            # POST request to the live API
├── model_card.md
├── pytest.ini
├── requirements.txt
└── setup.py

## Data

The raw file ``data/census.csv`` has leading spaces in its column names and values. These spaces were removed and the result was saved as ``data/census_clean.csv``. Column names were not changed.

## Training
Run:
```
python starter/train_model.py
```

The script splits the data and trains the model. It prints the precision, recall and F1 score. Furthermore, it writes per-slice metrics to slice_output.txt, and saves the model, encoder and label binarizer to ``model/``.

## API
| Method |	Path |	Description |
|--------|-------|--------------|
| GET	| ``/`` |	Returns a welcome message |
| POST	| ``/predict`` |	Returns the predicted income class

## Testing
The tests include unit tests for the model functions as well as API tests.

## CI/CD
CI: GitHub Actions runs flake8 and pytest with Python 3.13 on every push and pull request to master.

Continuous deployment: Render's auto-deploy is turned off. APp is only deployed by the deploy job in GitHub actions, which depends on the test job.