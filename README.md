# Student Performance Predictor

A complete Streamlit learning-analytics app built with the provided 200-row student dataset. It explores the data, visualizes academic patterns, evaluates a saved **Linear Regression** pipeline, and predicts a score estimate from a new student's profile.

> This is an educational regression project. A prediction is an estimate based on a small dataset, not a guarantee of a student's actual result.

## Project contents

```text
student-performance-predictor/
├── app.py
├── train_model.py
├── student_performance.csv
├── reference_analysis.ipynb
├── requirements.txt
├── .python-version
├── README.md
└── models/
    ├── student_score_pipeline.joblib
    ├── evaluation_metrics.json
    ├── test_predictions.csv
    ├── feature_coefficients.csv
    └── training_diagnostics.png
```

The notebook is included as the original analysis reference. Its useful steps (data inspection, missing-value checks, EDA, a Linear Regression baseline, and test-set evaluation) have been carried into the app and training script. The new implementation fixes two issues in the notebook: it excludes `Student_ID` and fits all imputation/encoding steps only on the training data.

## Dataset and model

- Target: `Final_Exam_Score`
- Model inputs: `Gender`, `Age`, `Study_Hours`, `Attendance`, `Previous_Score`, `Sleep_Hours`, `Internet_Hours`, `Assignment_Score`, `Extra_Classes`, and `Family_Support`
- Excluded: `Student_ID` (identifier, not a predictor)
- Split: 80% training / 20% testing, `random_state=42`
- Preprocessing: numeric median imputation; categorical most-frequent imputation and one-hot encoding; all steps are inside a Scikit-learn `Pipeline` with a `ColumnTransformer`
- Evaluation: MAE, MSE, RMSE, and R² calculated on the held-out test split

The uploaded CSV has 200 records and 12 columns. In the supplied copy, Attendance, Sleep_Hours, and Assignment_Score each have four missing values. The training pipeline fills missing input values from training-set data; it drops duplicate rows if present and excludes any row with a missing target because there is no known score to train against.

## Run on Windows

### 1. Install Python

Install Python 3.11 from [python.org](https://www.python.org/downloads/). On the installer screen, enable **Add Python to PATH**. Open a new PowerShell window afterward.

### 2. Open the project folder

```powershell
cd path\to\student-performance-predictor
```

For example, if you extracted the ZIP to Downloads:

```powershell
cd "$HOME\Downloads\student-performance-predictor"
```

### 3. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, run this once for the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

For Windows Command Prompt instead of PowerShell:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
```

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Train and save the model

The supplied project ZIP includes the trained pipeline and test evaluation files, so training is not required just to launch it. To reproduce or refresh those files from the CSV:

```powershell
python train_model.py
```

The script creates `models/` automatically and writes the trained pipeline, calculated metrics, held-out predictions, coefficients, and a diagnostic image there. The Streamlit app loads the saved pipeline; it does not retrain each time it starts.

### 6. Launch Streamlit

```powershell
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Press `Ctrl+C` in the terminal to stop the app.

## Publish on Streamlit Community Cloud

1. Create a GitHub repository and add the contents of this project folder.
2. Make sure the repository includes `student_performance.csv`, `requirements.txt`, and the files in `models/`. Do not upload `.venv/`.
3. Push the files to GitHub.
4. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub and choose **Create app**.
5. Select the repository, branch, and `app.py` as the main file path.
6. In the app's advanced settings, select Python 3.11 to match `.python-version`, then deploy.
7. If you retrain the model, commit the updated files from `models/` along with the source and dataset before redeploying.

The app reads the CSV and saved model files relative to `app.py`; it does not use a path from the developer's computer. Include the `models/` folder in GitHub so Streamlit Cloud can load the pre-trained model.

## How Linear Regression works here

Linear Regression learns a weighted relationship between the student inputs and final exam score. During training, it chooses coefficients that minimize the squared difference between known scores and its estimates. Numeric inputs stay numeric; categories are converted to one-hot columns. The coefficients show how the estimate changes with an input while the other inputs are held constant.

The training script first removes duplicate records and rows without a target score, then splits the remaining examples into training and test groups. The preprocessing steps are fitted using only the training group, which avoids letting test information influence the model. It evaluates predictions on the held-out test group and saves the resulting metrics and full preprocessing/model pipeline.

When someone submits the prediction form, Streamlit builds a one-row table with the same feature names used during training. The saved pipeline applies the same imputers and category encoding, then Linear Regression returns the score estimate. The app also provides rule-based study suggestions from entered values; these suggestions are not generated by the model.

## Troubleshooting

### `python`, `py`, or `streamlit` is not recognized

- Reinstall Python 3.11 with **Add Python to PATH** enabled, then open a fresh terminal.
- Activate `.venv` before installing packages or launching Streamlit.
- If the `streamlit` command is unavailable, try `python -m streamlit run app.py`.

### Dependency installation fails

- Confirm that the virtual environment is active and that Python is 3.11.
- Update pip with `python -m pip install --upgrade pip`, then retry `python -m pip install -r requirements.txt`.
- On Windows, close other Python processes if a package file is locked, then reopen PowerShell and retry.

### Dataset not found or required columns are missing

- Keep `student_performance.csv` beside `app.py` and `train_model.py`.
- Do not rename or remove the required columns listed in the dataset/model section.
- If using another CSV, replace the file with the same column names and UTF-8-compatible CSV format.

### Saved model files are missing or do not match the dataset

- Run `python train_model.py` from this folder using the same environment as the app.
- Confirm that the new files appeared under `models/`, then restart with `streamlit run app.py`.
- For Community Cloud, commit the regenerated `models/` files and redeploy so the hosted app receives them.

### Streamlit Community Cloud cannot load the model

- Confirm `.python-version` and Cloud advanced settings both specify Python 3.11.
- Ensure `requirements.txt` is committed and contains Scikit-learn.
- Recreate the model files with the requirements installed in the same Python version, then commit those files and redeploy.
