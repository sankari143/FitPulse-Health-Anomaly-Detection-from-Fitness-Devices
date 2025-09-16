# FitPulse-Health-Anomaly-Detection-from-Fitness-Devices
# 🏃 FitPulse – Health Anomaly Detection from Fitness Devices  

FitPulse is a Python-based project that analyzes **fitness tracker data** (heart rate, steps, sleep) to detect anomalies such as abnormal spikes, unusual patterns, or irregular activity. It helps demonstrate **anomaly detection** in healthcare IoT data using machine learning and visualization.  

---

## 🚀 Features  
- 📊 **Data Processing**: Reads CSV/JSON data from fitness trackers.  
- 🧹 **Preprocessing**: Cleans timestamps, fills missing values, resamples to 1-minute intervals.  
- ⚙️ **Anomaly Detection**:  
  - Threshold-based (too high/low heart rate)  
  - Residual-based (deviation from rolling mean)  
  - Clustering (DBSCAN outlier detection)  
- 📉 **Visualization**: Plots heart rate with anomalies highlighted.  
- 📂 **Outputs**:  
  - `processed.csv` → data with anomaly labels  
  - `anomalies_plot.png` → visualization of detected anomalies  
- 🌐 **Dashboard (optional)**: Streamlit app for interactive analysis.  

---

## 📂 Project Structure  
