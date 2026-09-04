# Wine Quality ML API

A production-grade machine learning API that predicts wine quality from chemical properties.
The model classifies wines as **High Quality** or **Low Quality** and returns a confidence score
with every prediction. Built with FastAPI, deployed on Render, and backed by PostgreSQL for
real-time prediction logging.

**Live API:** https://wine-quality-api-9jan.onrender.com/docs  
**Live Dashboard:** https://6a9aa374bdf0e0f93fb64e8a--fanciful-bublanina-b3f50f.netlify.app  
**Dashboard Repository:** https://github.com/Krish-Script/wine-dashboard

---

## Architecture
```
**Client Request** 
        ↓
**FastAPI** *(Hosted on Render)*
        ↓
**XGBoost Classifier**
        ↓
**Prediction + Confidence Score**
        ↓
**PostgreSQL** *(Prediction Logged)*
        ↓
**Response Returned to Client**
```


---

## Technical Decisions

**Why Binary Classification?**
The original UCI dataset rates wines from 3–8, but 82% of samples are rated 5 or 6.
A multiclass model would achieve high accuracy by lazily predicting the majority class.
I reframed the problem into binary classification — Low (3–5) vs High (6–8) — producing
a balanced 744/855 split and meaningful F1 scores across both classes.

**Why XGBoost?**
XGBoost is highly efficient on structured tabular data like chemical readings.
It achieved 79% F1 score on the balanced dataset without extensive hyperparameter tuning,
and its native JSON serialization format ensures consistent behavior across environments.

**Why these features?**
After correlation analysis, three features showed near-zero correlation with quality:
residual sugar (0.013), free sulfur dioxide (-0.050), and pH (-0.057). Dropping them
reduced noise without degrading model performance.

---

## API Endpoints

### `GET /`
Health check — confirms the API is running.

### `POST /predict`
Takes wine chemical properties, returns prediction and confidence score.

**Request body:**
```json
{
  "fixed_acidity": 8.1,
  "volatile_acidity": 0.28,
  "citric_acid": 0.40,
  "chlorides": 0.068,
  "total_sulfur_dioxide": 30.0,
  "density": 0.9950,
  "sulphates": 0.82,
  "alcohol": 12.5
}
```

**Response:**
```json
{
  "prediction": "High Quality",
  "confidence": 99.8
}
```

### `GET /predictions/history`
Returns the last 20 predictions with inputs, results, and timestamps.

---

## Engineering Challenges

**Environment version mismatch** — My Anaconda environment used sklearn 1.9.0 while
VS Code's system Python used 1.8.0. Loading a scaler saved on one version in another
can silently produce wrong predictions. Resolved by upgrading sklearn across environments
and switching model serialization to XGBoost's native JSON format.

**Deployment build failure** — Running `pip freeze` inside Anaconda dumped Windows-specific
package paths into `requirements.txt` that don't exist on Render's Linux servers.
Resolved by replacing it with a minimal, platform-agnostic requirements file.

---

## Running Locally

```bash
git clone https://github.com/Krish-Script/wine-quality-api.git
cd wine-quality-api
pip install -r requirements.txt
```

Create a `.env` file:
```
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/winedb
```


Start the server:
```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/docs`

---

## Stack

- **Model** — XGBoost Classifier
- **API** — FastAPI + Pydantic
- **Database** — PostgreSQL + SQLAlchemy
- **Deployment** — Render
- **Dataset** — UCI Red Wine Quality