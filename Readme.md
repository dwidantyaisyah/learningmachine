# AIS Collection Dashboard ✨

## Setup Environment

This project demonstrates advanced data analysis techniques using the Bike Sharing dataset. It includes data manipulation, exploratory data analysis (EDA), RFM (Recency, Frequency, Monetary) analysis, and visualizations. The project leverages multiple Python libraries for efficient and effective data analysis.

## Project Overview

The project uses the following datasets:
1. `day.csv`: Aggregated daily data for bike sharing.
2. `hour.csv`: Hourly data for bike sharing.
3. `Copy_of_Proyek_Analisis_Data.ipynb`: Jupyter Notebook containing code and analysis steps.

### Key Features
- **Exploratory Data Analysis (EDA)**: Understanding patterns and trends in bike rentals based on weather, seasons, working days, and holidays.
- **RFM Analysis**: Grouping data based on recency, frequency, and monetary value using segments like seasons, weather conditions, and working days.
- **Visualizations**: Creating insightful visualizations to highlight rental patterns and RFM segmentation.
- **Data Preparation**: Handling missing values, type conversion, and data transformation.

---

## Requirements

This project is built using Python 3.7+ and requires the following libraries. Install them using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
python3 -m venv env
source env/bin/activate
pip install -r requirement.txt
```

### Run steamlit app

```
streamlit run ./dashboard/dashboard.py
```
