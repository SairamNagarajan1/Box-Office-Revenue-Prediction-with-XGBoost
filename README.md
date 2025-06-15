# 🎬 Box Office Revenue Prediction with XGBoost

This project aims to predict the **domestic revenue** of movies using machine learning techniques, specifically XGBoost Regression. We explore and clean the data, visualize key metrics, encode categorical variables, normalize data, and build a predictive model.

## 📂 Dataset

Dataset Source: [boxoffice.csv](boxoffice.csv)  
(*The dataset includes features like title, MPAA rating, distributor, release details, and revenue.*)

## 📊 Features Used

- `MPAA`, `distributor`, `release_days`, `opening_theaters`, `genres`
- Transformed using Label Encoding and One-Hot Encoding
- Log transformation applied to skewed numeric features

## 🔧 Preprocessing Steps

- Removed redundant revenue columns (`world_revenue`, `opening_revenue`)
- Filled missing values using mode
- Removed special characters (`$`, `,`) and converted numeric columns
- One-hot encoded `genres` using `CountVectorizer` with rare genre filtering
- Log transformed skewed data
- Standardized numerical features using `StandardScaler`

## 📈 Model

- Model: `XGBoost Regressor`
- Training Size: 90%
- Validation Size: 10%
- Evaluation Metric: **Mean Absolute Error (MAE)**

### 🧠 Results

- **Training MAE**: ~0.21  
- **Validation MAE**: ~0.63  
- Some overfitting observed; can be improved with hyperparameter tuning.

## 📎 Dependencies

- Python 3.7+
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- xgboost
