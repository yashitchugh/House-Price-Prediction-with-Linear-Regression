# 🏠 House Price Prediction with Linear Regression

This project builds a **Linear Regression** model using Python and scikit-learn to predict housing prices based on various features like area, location, and amenities.

---

## 📁 Dataset

The dataset contains details about residential properties, including:

- `area`: Size in square feet
- `mainroad`, `guestroom`, `hotwaterheating`, `airconditioning`, `prefarea`: Binary features (Yes/No)
- `price`: Target variable

---

## 🧹 Data Preprocessing

✅ Steps taken:

- Converted binary categorical values (`yes`/`no`) into `1`/`0` for key columns  
- Handled train-test split with `train_test_split()`

```python
df[["mainroad", "guestroom", "hotwaterheating", "airconditioning", "prefarea"]] = \
    df[["mainroad", "guestroom", "hotwaterheating", "airconditioning", "prefarea"]].replace({'yes': 1, 'no': 0})
````

---

## 🤖 Model Training

We trained a **Linear Regression** model using scikit-learn:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

---

## 📊 Evaluation

Evaluated the model using:

* **MAE (Mean Absolute Error)**
* **MSE (Mean Squared Error)**
* **R² Score**

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

---

## 📈 Visualization

If using one feature (e.g., `area`), we plotted the regression line:

```python
plt.scatter(X_test['area'], y_test, color='blue')
plt.plot(X_test['area'], model.predict(X_test), color='red')
```

---

## 🔍 Coefficients Interpretation

Each coefficient shows the effect of that feature on house price. Positive = price increases, negative = decreases.

```python
pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/house-price-regression.git
cd house-price-regression

# Install dependencies
pip install pandas scikit-learn matplotlib

# Run the script
python main.py  # or Jupyter Notebook
```

---

## 📌 Conclusion

This project is a hands-on application of linear regression for real-world data, showcasing data cleaning, model building, evaluation, and interpretation — all in a clean, reproducible pipeline.

---
