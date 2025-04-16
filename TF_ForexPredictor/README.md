# 💹 TF_ForexPredictor

A time series prediction model that forecasts foreign exchange prices using LSTM networks in TensorFlow. Historical data is processed to predict short-term price movement patterns.

---

## 🎯 Objectives

- Build and train LSTM models for FX forecasting
- Preprocess raw forex data (scaling, windowing)
- Visualize prediction accuracy over time

---

## 📊 Main packages used:
 - tensorflow
 - pandas
 - numpy
 - matplotlib
 - sklearn

**Note: Normalize and reshape data into supervised learning format before training.**

---

## 🧠 Notes
 - Implemented early stopping to prevent overfitting
 - Supports prediction horizon configuration
 - Plots training loss and prediction overlays

---

## 📁 Folder Structure

```bash
TF_ForexPredictor/
├── createSavedDataset.ipynb   
├── customRatioLayerDefinition.ipynb                 
├── buildAndTrainModel.ipynb
├── mySavedModel.keras                  
└── README.md                
