# 📜 開發紀錄檔 (log.md)

## v0.1 初始化專案
- 建立 `README.md`
  - 撰寫專案簡介與功能說明
  - 說明資料生成方式（含斜率、截距、噪聲設定）
  - 加入教學用途與 Demo 網頁連結

## v0.2 建立 Streamlit 主程式 (`app.py`)
- 引入必要套件：
  - `streamlit`, `numpy`, `matplotlib`
  - `sklearn.linear_model.LinearRegression`
  - `sklearn.model_selection.train_test_split`
  - `sklearn.metrics` (MSE, R²)
- 實作功能：
  - **側邊欄控制項**
    - 資料數量 (100 ~ 5000)
    - 真實斜率 a
    - 噪聲強度
  - 固定截距 b = 50
  - 資料生成（均勻分布的 x，加上高斯噪聲）
  - 訓練/驗證資料切分 (80/20)
  - 建立並訓練 `LinearRegression` 模型
  - 計算並顯示評估指標 (R², MSE, RMSE)
  - 視覺化：
    - 訓練資料散點圖
    - 擬合直線

## v0.3 改進方向（待辦事項）
- [ ] 在側邊欄開放 **截距 b** 調整
- [ ] 新增 **Noise 分布類型選擇** (Uniform / Gaussian)
- [ ] 顯示 **驗證集散點與擬合結果**
- [ ] 加入 **模型殘差圖**，觀察模型誤差分布
- [ ] 部署到 Streamlit Cloud，提供即時 Demo

---
