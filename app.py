import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- Sidebar controls ---
st.sidebar.title("🔧 參數調整")

n_samples = st.sidebar.slider("資料數量", min_value=100, max_value=5000, value=1000, step=100)
true_slope = st.sidebar.slider("真實資料斜率 a（y = ax + b + noise）", min_value=0.0, max_value=20.0, value=10.0, step=0.5)
noise_level = st.sidebar.slider("噪聲強度（Noise Level）", min_value=0.0, max_value=20.0, value=5.0, step=0.5)

# --- 固定截距（你也可以開放調整）---
intercept = 50.0

# --- Generate data ---
np.random.seed(42)
x = np.random.uniform(0, 10, n_samples)
noise = np.random.normal(0, noise_level, n_samples)
y = true_slope * x + intercept + noise

# --- Split data ---
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)
x_train_reshaped = x_train.reshape(-1, 1)
x_valid_reshaped = x_valid.reshape(-1, 1)

# --- Train model ---
model = LinearRegression()
model.fit(x_train_reshaped, y_train)

# --- Predictions ---
y_train_pred = model.predict(x_train_reshaped)
y_valid_pred = model.predict(x_valid_reshaped)

# --- Evaluation metrics ---
def get_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return r2, mse, rmse

train_r2, train_mse, train_rmse = get_metrics(y_train, y_train_pred)
valid_r2, valid_mse, valid_rmse = get_metrics(y_valid, y_valid_pred)

# --- Display model results ---
st.title("📈 線性回歸模擬器")

st.markdown("### 🎯 模型擬合結果")
st.write(f"**實際生成資料的斜率 (a):** {true_slope}")
st.write(f"**模型學到的斜率 (â):** {model.coef_[0]:.4f}")
st.write(f"**模型學到的截距 (b̂):** {model.intercept_:.4f}")

st.markdown("### 📊 評估指標")

col1, col2 = st.columns(2)

with col1:
    st.subheader("訓練集")
    st.write(f"R²: {train_r2:.4f}")
    st.write(f"MSE: {train_mse:.4f}")
    st.write(f"RMSE: {train_rmse:.4f}")

with col2:
    st.subheader("驗證集")
    st.write(f"R²: {valid_r2:.4f}")
    st.write(f"MSE: {valid_mse:.4f}")
    st.write(f"RMSE: {valid_rmse:.4f}")

# --- Plot ---
st.markdown("### 🧪 訓練資料與擬合線")

fig, ax = plt.subplots()
ax.scatter(x_train, y_train, alpha=0.3, label='訓練資料')
sorted_idx = np.argsort(x_train)
ax.plot(x_train[sorted_idx], y_train_pred[sorted_idx], color='red', linewidth=2, label='擬合直線')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('線性回歸擬合（訓練集）')
ax.legend()
st.pyplot(fig)
