# Credit Card Fraud Detection — Deployment Package

This folder is ready to push to GitHub and deploy on Vercel.

## ⚠️ Before uploading to GitHub — 1 required step

This package does NOT include your trained model files, since they must come
from your own Kaggle notebook. You must add them yourself:

1. In your Kaggle notebook, make sure you've run:
   ```python
   import joblib
   joblib.dump(rf_smote, 'fraud_model.pkl')
   joblib.dump(scaler, 'scaler.pkl')
   ```
2. Go to the **Output** panel (right sidebar of your Kaggle notebook)
3. Download both `fraud_model.pkl` and `scaler.pkl`
4. Place both files inside this project's `model/` folder, so it looks like:
   ```
   fraud-app/
   ├── api/predict.py
   ├── model/
   │   ├── fraud_model.pkl   <-- add this
   │   └── scaler.pkl        <-- add this
   ├── index.html
   ├── requirements.txt
   ├── vercel.json
   └── README.md
   ```

## Upload to GitHub

1. Create a new repository on GitHub (e.g. `fraud-app`)
2. Upload this entire folder — keep the `api/` and `model/` subfolders intact,
   don't flatten the structure
3. Commit

## Deploy on Vercel

1. Go to https://vercel.com → **Add New Project**
2. Import your `fraud-app` GitHub repo
3. Framework preset: **Other**
4. Click **Deploy**
5. Wait for the build to finish — you'll get a live URL

## Test it

Open your Vercel URL. You should see the fraud detection form. Submit a
transaction amount and time to get a live prediction.

## If the build fails

The most likely cause is `fraud_model.pkl` being too large for Vercel's
free-tier function size limit (~50MB). Check the file size on your computer —
if it's large, retrain with fewer trees in Kaggle:

```python
from sklearn.ensemble import RandomForestClassifier
rf_smote_small = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
rf_smote_small.fit(X_train_sm, y_train_sm)
joblib.dump(rf_smote_small, 'fraud_model.pkl')
```

Fewer trees = smaller file, still solid performance, and now guaranteed to
fit within Vercel's limits.

## Notes

- V1–V28 are anonymized PCA features from the original dataset with no
  real-world meaning, so the live demo simulates them and only takes
  Amount and Time as real user inputs (this is disclosed in the interface
  itself).
- API and interface are served from the same Vercel deployment (same
  domain), so no CORS configuration is needed.
