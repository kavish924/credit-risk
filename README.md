# 🏦 Credit Risk Prediction — MLOps Pipeline

> **Predict whether a loan applicant will default** using Machine Learning, with a full production-grade MLOps pipeline.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green?logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-orange?logo=xgboost)
![MLflow](https://img.shields.io/badge/MLflow-2.19-blue?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-black?logo=github-actions)

---

## 📖 What Does This Project Do?

In simple terms, **banks need to decide whether to approve or reject a loan application**. This project builds an AI model that:

1. **Looks at an applicant's data** (income, loan amount, age, employment history, etc.)
2. **Predicts the chance of default** (i.e., the person failing to repay the loan)
3. **Returns a risk score** — LOW, MEDIUM, or HIGH

But it doesn't stop at just building a model — it implements the **full MLOps lifecycle**:

- ✅ Automated data loading & validation
- ✅ Feature engineering & preprocessing
- ✅ Model training with experiment tracking (MLflow)
- ✅ REST API for real-time predictions (FastAPI)
- ✅ Web dashboard for manual predictions
- ✅ Data drift monitoring (Evidently AI)
- ✅ CI/CD pipelines (GitHub Actions)
- ✅ Docker containerization
- ✅ Airflow DAG for weekly retraining

---

## 🏗️ Project Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                              │
│  Kaggle Dataset → load_data.py → validate.py → preprocessing  │
└────────────────────┬───────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────────┐
│                     ML TRAINING LAYER                          │
│  Logistic Regression / Random Forest / XGBoost                 │
│  SMOTE Resampling → 5-Fold CV → Threshold Tuning              │
│  MLflow Experiment Tracking                                    │
└────────────────────┬───────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────────┐
│                     SERVING LAYER                              │
│  FastAPI REST API ──► /predict  /predict/batch  /health        │
│  Frontend Web Dashboard (HTML/CSS/JS)                          │
│  Prometheus Metrics at /metrics                                │
└────────────────────┬───────────────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────────────────┐
│                   MONITORING & OPS                             │
│  Evidently AI Drift Reports                                    │
│  GitHub Actions CI/CD                                          │
│  Docker + Docker Compose                                       │
│  Airflow Weekly Retraining DAG                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## 📂 Folder Structure

```
credit-risk/
│
├── data/                        # Data directory
│   ├── raw/                     # Raw CSV files from Kaggle (not tracked in git)
│   └── processed/               # Preprocessed data & artifacts
│       ├── preprocessor.pkl     # Fitted sklearn preprocessor pipeline
│       └── feature_names.pkl    # List of feature names after preprocessing
│
├── src/                         # Core source code
│   ├── config.py                # Environment configuration (paths, MLflow URI)
│   ├── data/
│   │   ├── load_data.py         # Load raw CSV files with basic validation
│   │   ├── preprocessing.py     # Feature engineering + sklearn pipeline
│   │   └── validate.py          # Data quality checks
│   └── ml/
│       ├── train.py             # Train 3 models with MLflow tracking
│       └── evaluate.py          # Evaluate best model + generate reports
│
├── api/                         # REST API
│   ├── app.py                   # FastAPI application with all endpoints
│   ├── model_loader.py          # Loads model from MLflow or local file
│   └── schemas.py               # Pydantic request/response schemas
│
├── frontend/                    # Web dashboard
│   ├── index.html               # Main HTML page
│   ├── style.css                # Styling
│   └── app.js                   # Frontend logic (form → API → result)
│
├── monitoring/
│   └── drift.py                 # Evidently AI data drift detection
│
├── airflow/dags/
│   └── credit_risk_dag.py       # Airflow DAG for automated retraining
│
├── tests/                       # Unit tests
│   ├── test_api.py              # API endpoint tests (mocked model)
│   └── test_preprocessing.py    # Preprocessing pipeline tests
│
├── notebooks/
│   └── eda.ipynb                # Exploratory Data Analysis notebook
│
├── .github/workflows/           # CI/CD pipelines
│   ├── ci.yml                   # Lint → Test → Docker build
│   └── cd.yml                   # Build & deploy Docker image
│
├── Dockerfile                   # Multi-stage Docker build
├── docker-compose.yml           # MLflow + API services
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (not tracked in git)
└── .gitignore                   # Files excluded from git
```

---

## 🚀 Step-by-Step Setup Guide

### Prerequisites

Make sure you have these installed on your computer:

| Tool | Version | What it's for |
|------|---------|---------------|
| **Python** | 3.10+ | Running the code |
| **pip** | Latest | Installing Python packages |
| **Git** | Any | Cloning the repository |
| **Docker** *(optional)* | Latest | Running the app in containers |

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/kavish924/credit-risk.git
cd credit-risk
```

---

### Step 2: Create a Virtual Environment

A virtual environment keeps this project's packages separate from your other Python projects.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You'll see `(venv)` at the start of your terminal — that means it's working.

---

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs all the libraries the project needs (pandas, scikit-learn, XGBoost, FastAPI, etc.).

> **Note:** You may also need `imbalanced-learn` for SMOTE resampling:
> ```bash
> pip install imbalanced-learn
> ```

---

### Step 4: Download the Dataset

This project uses the **Home Credit Default Risk** dataset from Kaggle.

1. Go to: https://www.kaggle.com/c/home-credit-default-risk/data
2. Download `application_train.csv` and `application_test.csv`
3. Place them in the `data/raw/` folder:

```
data/
└── raw/
    ├── application_train.csv    ← Put here
    └── application_test.csv     ← Put here
```

> **Tip:** You can also use the Kaggle CLI:
> ```bash
> pip install kaggle
> kaggle competitions download -c home-credit-default-risk -p data/raw/
> ```

---

### Step 5: Set Up Environment Variables

Create a `.env` file in the project root (or edit the existing one):

```env
APP_ENV=development
PORT=8000
DATA_PATH=data/
MODEL_PATH=models/
MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

---

### Step 6: Run Data Preprocessing

This step does the heavy lifting of cleaning and transforming the raw data:

```bash
python -m src.data.preprocessing
```

**What happens behind the scenes:**
1. Loads the raw CSV file (~307k rows)
2. Splits into 80% training / 20% validation
3. Drops columns with >50% missing values
4. Creates new features (debt-to-income ratio, age, credit term, etc.)
5. Applies median imputation for missing numbers
6. One-hot encodes categorical features (like gender, contract type)
7. Scales numeric features using RobustScaler
8. Saves everything to `data/processed/`

---

### Step 7: Train the Models

```bash
python -m src.ml.train
```

**What happens:**
1. Trains **3 models** — Logistic Regression, Random Forest, and XGBoost
2. Uses **SMOTE** to handle class imbalance (only ~8% of loans default)
3. Runs **5-fold cross-validation** on each model
4. Finds the **best classification threshold** using the Precision-Recall curve
5. Logs everything to **MLflow** (or local files if MLflow isn't running)
6. Saves the best model as `models/best_model.pkl`

**You can customize training:**
```bash
python -m src.ml.train --n-estimators 500 --max-depth 6 --learning-rate 0.03
```

| Argument | Default | What it controls |
|----------|---------|-----------------|
| `--n-estimators` | 500 | Number of trees in XGBoost |
| `--max-depth` | 4 | How deep each tree can grow |
| `--learning-rate` | 0.03 | How fast the model learns |
| `--sampling` | smote | Resampling strategy: `smote`, `undersample`, `combined`, `none` |

---

### Step 8: Evaluate the Best Model

```bash
python -m src.ml.evaluate
```

**What happens:**
- Prints a classification report (precision, recall, F1-score)
- Generates an ROC curve + confusion matrix plot → `monitoring/reports/roc_confusion.png`
- Generates SHAP feature importance plot → `monitoring/reports/shap_summary.png`

---

### Step 9: Start the API Server

```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

The API is now running! Open your browser and go to:

| URL | What you'll see |
|-----|-----------------|
| http://localhost:8000 | 🖥️ **Web Dashboard** — Enter applicant details and get predictions |
| http://localhost:8000/docs | 📄 **Swagger UI** — Interactive API documentation |
| http://localhost:8000/health | 💚 Health check endpoint |
| http://localhost:8000/metrics | 📊 Prometheus metrics |

---

### Step 10: Make a Prediction

#### Option A: Use the Web Dashboard

Open http://localhost:8000 in your browser. Fill in the form with applicant details and click **"Predict Risk"**. You'll see:
- **Default Probability** (e.g., 0.15 = 15% chance of default)
- **Risk Label** (LOW / MEDIUM / HIGH)
- **Risk Score** (0–1000, higher = safer)

#### Option B: Use the API Directly

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "AMT_CREDIT": 540000,
    "AMT_INCOME_TOTAL": 135000,
    "AMT_ANNUITY": 26000,
    "DAYS_BIRTH": -12000,
    "DAYS_EMPLOYED": -3000,
    "NAME_CONTRACT_TYPE": "Cash loans",
    "CODE_GENDER": "M",
    "FLAG_OWN_CAR": "N",
    "FLAG_OWN_REALTY": "Y"
  }'
```

**Response:**
```json
{
  "default_probability": 0.1523,
  "risk_label": "LOW",
  "risk_score": 848,
  "model_version": "local",
  "model_name": "XGBClassifier"
}
```

#### Option C: Batch Predictions (up to 100 at once)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "applications": [
      {"AMT_CREDIT": 540000, "AMT_INCOME_TOTAL": 135000, "AMT_ANNUITY": 26000, "DAYS_BIRTH": -12000},
      {"AMT_CREDIT": 200000, "AMT_INCOME_TOTAL": 80000, "AMT_ANNUITY": 12000, "DAYS_BIRTH": -8000}
    ]
  }'
```

---

## 🐳 Docker Deployment

### Run with Docker Compose (Recommended)

This starts both the **MLflow server** and the **Credit Risk API** together:

```bash
docker-compose up --build
```

| Service | URL | Description |
|---------|-----|-------------|
| Credit Risk API | http://localhost:8000 | Prediction API + Dashboard |
| MLflow UI | http://localhost:5000 | Experiment tracking dashboard |

### Run Standalone with Docker

```bash
docker build -t credit-risk-api .
docker run -p 8000:8000 -v ./models:/app/models -v ./data:/app/data credit-risk-api
```

---

## 📊 Monitoring & Drift Detection

Over time, the data your model sees in production might change (this is called **data drift**). This project uses **Evidently AI** to detect drift.

### Run a Drift Report

```bash
python -m monitoring.drift --ref-size 5000 --curr-size 1000
```

**What happens:**
1. Takes the first 5000 training rows as the **reference** (what the model was trained on)
2. Takes the last 1000 validation rows as **current data** (simulating new production data)
3. Compares them statistically for each feature
4. Generates an HTML report → `monitoring/reports/drift_report_latest.html`

Open the HTML file in your browser to see a visual drift report.

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run only preprocessing tests
pytest tests/test_preprocessing.py -v

# Run only API tests
pytest tests/test_api.py -v
```

The API tests use a **mock model** so you don't need the actual trained model to run them.

---

## 🔄 CI/CD Pipeline (GitHub Actions)

The project includes two automated workflows:

### CI Pipeline (`.github/workflows/ci.yml`)

**Triggered on:** Every push to `main` or `develop`, and every pull request.

```
Lint (ruff) → Unit Tests → Docker Build → Smoke Test
```

### CD Pipeline (`.github/workflows/cd.yml`)

**Triggered on:** Every push to `main`.

```
Build Docker Image → Push to Docker Hub → Deploy via SSH
```

> **Setup required:** Add these secrets in your GitHub repo settings (`Settings → Secrets → Actions`):
> - `DOCKER_USERNAME` — Your Docker Hub username
> - `DOCKER_PASSWORD` — Your Docker Hub password/token
> - `DEPLOY_HOST`, `DEPLOY_USER`, `DEPLOY_SSH_KEY` — *(Optional)* For SSH deployment

---

## 🌬️ Airflow (Automated Retraining)

The project includes an Airflow DAG that runs the **full pipeline every Sunday**:

```
Ingest → Validate → Preprocess → Train → Evaluate → Monitor → Notify
```

To use it, copy the DAG file to your Airflow dags folder:

```bash
cp airflow/dags/credit_risk_dag.py ~/airflow/dags/
```

The DAG includes a **quality gate**: if the model's ROC-AUC drops below 0.70, the pipeline stops and won't deploy a bad model.

---

## 🧠 How the ML Model Works (Simplified)

1. **Problem:** Given a person's financial data, predict if they'll fail to repay a loan (binary: yes/no)

2. **Challenge:** Only ~8% of people default — the data is very imbalanced. We handle this with:
   - **SMOTE** — Creates synthetic examples of the minority class
   - **Class weights** — Tells the model to pay more attention to defaults
   - **Threshold tuning** — Instead of using 0.5 as the cutoff, we find the optimal threshold

3. **Models trained:**
   - 📈 **Logistic Regression** — Simple, interpretable baseline
   - 🌲 **Random Forest** — Ensemble of decision trees
   - ⚡ **XGBoost** — Gradient boosted trees (usually the best)

4. **Evaluation metrics:**
   - **ROC-AUC** — How well the model separates defaulters from non-defaulters
   - **Average Precision** — Focuses on how well we identify the rare defaulters
   - **F1-Score** — Balance between precision and recall

5. **Feature engineering** adds smart derived features:
   - `DEBT_TO_INCOME` = Loan Amount / Income
   - `ANNUITY_TO_INCOME` = Monthly Payment / Income
   - `CREDIT_TERM` = Loan Amount / Monthly Payment
   - `AGE_YEARS` = Age in years (from days)

---

## 📋 API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web dashboard (frontend UI) |
| `GET` | `/health` | Health check — is the model loaded? |
| `GET` | `/docs` | Swagger API documentation |
| `GET` | `/model/info` | Model metadata (name, version) |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/predict` | Single loan prediction |
| `POST` | `/predict/batch` | Batch predictions (up to 100) |

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **ML/Data** | pandas, scikit-learn, XGBoost, imbalanced-learn, SHAP |
| **Experiment Tracking** | MLflow |
| **API** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | HTML, CSS, JavaScript |
| **Monitoring** | Evidently AI, Prometheus |
| **CI/CD** | GitHub Actions |
| **Containers** | Docker, Docker Compose |
| **Orchestration** | Apache Airflow |
| **Testing** | pytest, httpx |

---

## 📜 License

This project is open source and available for educational and portfolio purposes.

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m "Add amazing feature"`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/kavish924">kavish924</a>
</p>
