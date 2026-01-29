# 🏠 SmartHome Valuator API

Backend API for house price prediction in Semarang using Machine Learning. Built with FastAPI and uses ensemble models to generate accurate predictions.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Description

SmartHome Valuator is a REST API that predicts property prices in Semarang based on various features such as location, building area, number of rooms, and other amenities. This API uses an ensemble machine learning model trained on real property data to provide accurate price estimates.

### ✨ Key Features

- 🎯 **Accurate Predictions**: Uses ensemble models (XGBoost, Random Forest, Gradient Boosting)
- 🗺️ **Zone Mapping**: Automatic mapping of specific locations to Semarang zones
- 🔄 **Smart Imputation**: Handles missing data with intelligent defaults
- 📊 **Batch Prediction**: Predict multiple properties at once
- 🏷️ **Categorization**: Classifies properties from Economy to Luxury
- 📈 **Confidence Interval**: Provides prediction uncertainty estimates
- 🚀 **Auto Model Download**: Automatic model download from remote URL
- 📝 **API Documentation**: Interactive docs with Swagger UI

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **ML Models**: XGBoost, Random Forest, Gradient Boosting
- **Data Processing**: Pandas, NumPy
- **Model Serialization**: Pickle, Joblib
- **Server**: Uvicorn
- **Validation**: Pydantic

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Local Setup

1. **Clone repository**
```bash
git clone https://github.com/yourusername/smarthome-valuator.git
cd smarthome-valuator
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup environment variables**

Create a `.env` file in the root directory:
```env
MODEL_DOWNLOAD_URL=https://your-model-storage-url/smarthome_complete_pipeline.pkl
```

5. **Run the application**
```bash
python main.py
```

The API will run at `http://localhost:8000`

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Root Endpoint
```http
GET /
```
Displays basic API information and available endpoints.

**Response:**
```json
{
  "message": "SmartHome Price Predictor API",
  "version": "1.0.0",
  "status": "active",
  "endpoints": {
    "predict": "/predict",
    "batch_predict": "/batch-predict",
    "health": "/health",
    "docs": "/docs"
  }
}
```

#### 2. Health Check
```http
GET /health
```
Checks API and model health status.

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running",
  "timestamp": "2026-01-29T10:30:00",
  "models_loaded": true
}
```

#### 3. Single Price Prediction
```http
POST /predict
```

**Request Body:**
```json
{
  "lokasi": "Tembalang",
  "kamar_tidur": 3,
  "kamar_mandi": 2,
  "luas_tanah": "150",
  "luas_bangunan": "120",
  "carport": 1,
  "daya_listrik": 1300,
  "jumlah_lantai": 2,
  "kondisi_properti": "Bagus",
  "kondisi_perabotan": "Tidak Berperabot"
}
```

**Response:**
```json
{
  "predicted_price": 1250000000,
  "predicted_price_formatted": "Rp 1.250.000.000",
  "predicted_price_miliar": 1.25,
  "confidence_interval": "Rp 1.150.000.000 - Rp 1.350.000.000",
  "uncertainty": 51020408.16,
  "property_category": "Menengah Bawah",
  "zona": "Semarang Timur",
  "preprocessing_info": {
    "missing_fields": [],
    "zona_mapped": "Semarang Timur",
    "features_shape": [1, 45]
  }
}
```

#### 4. Batch Prediction
```http
POST /batch-predict
```

**Request Body:**
```json
{
  "properties": [
    {
      "lokasi": "Tembalang",
      "kamar_tidur": 3,
      "luas_bangunan": "120"
    },
    {
      "lokasi": "Simpang Lima",
      "kamar_tidur": 4,
      "luas_bangunan": "200"
    }
  ]
}
```

**Response:**
```json
{
  "total_properties": 2,
  "successful_predictions": 2,
  "failed_predictions": 0,
  "results": [...]
}
```

#### 5. Model Information
```http
GET /model-info
```
Displays information about the currently loaded model.

#### 6. Available Zones
```http
GET /zones
```
Displays list of available zones and locations.

**Response:**
```json
{
  "total_zones": 6,
  "zones": {
    "Semarang Barat": ["Ngaliyan", "Kalibanteng", "Tugu", ...],
    "Semarang Timur": ["Tembalang", "Banyumanik", "Pedurungan", ...],
    ...
  },
  "zone_names": ["Semarang Barat", "Semarang Timur", ...]
}
```

## 📝 Input Schema

### PropertyInput Fields

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `lokasi` | string | ✅ | Property location in Semarang | "Tembalang" |
| `kamar_tidur` | int/string | ❌ | Number of bedrooms | 3 |
| `kamar_mandi` | int/string | ❌ | Number of bathrooms | 2 |
| `luas_tanah` | int/string | ❌ | Land area (m²) | "150" |
| `luas_bangunan` | int/string | ❌ | Building area (m²) | "120" |
| `carport` | int/string | ❌ | Number of carports | 1 |
| `daya_listrik` | int/string | ❌ | Electrical power (VA) | 1300 |
| `jumlah_lantai` | int/string | ❌ | Number of floors | 2 |
| `kondisi_properti` | string | ❌ | Property condition | "Bagus" |
| `kondisi_perabotan` | string | ❌ | Furnishing condition | "Furnished" |

### Notes:
- ✅ = Required field
- ❌ = Optional field (will be imputed with default values)
- All numeric fields can be sent as strings with units (will be cleaned automatically)

## 🏷️ Property Categories

The API classifies properties into 6 categories:

| Category | Price Range |
|----------|-------------|
| Ekonomis (Economy) | < Rp 500 million |
| Menengah Bawah (Lower Middle) | Rp 500 million - Rp 1 billion |
| Menengah (Middle) | Rp 1 billion - Rp 2 billion |
| Menengah Atas (Upper Middle) | Rp 2 billion - Rp 3.5 billion |
| Premium | Rp 3.5 billion - Rp 6 billion |
| Luxury | > Rp 6 billion |

## 🗺️ Semarang Zones

The API recognizes 6 main zones in Semarang:

1. **Semarang Barat (West)**: Ngaliyan, Kalibanteng, Tugu, BSB City, etc.
2. **Semarang Timur (East)**: Tembalang, Banyumanik, Pedurungan, Tlogosari, etc.
3. **Semarang Utara (North)**: Tanah Mas, Panggung, Kuningan, etc.
4. **Semarang Selatan (South)**: Wonodri, Candisari, Pleburan, etc.
5. **Semarang Tengah (Central)**: Simpang Lima, Pemuda, Sekayu, etc.
6. **Semarang Lainnya (Others)**: Ungaran, Bawen, Bergas, etc.

## 🔧 Configuration

### Environment Variables

```env
# Model Configuration
MODEL_DOWNLOAD_URL=https://your-storage-url/model.pkl

# Server Configuration (optional)
HOST=0.0.0.0
PORT=8000
```

### Model File Structure

```
models/
└── smarthome_complete_pipeline.pkl
```

Model file must contain a dictionary with keys:
- `models`: Dictionary of ML models
- `ensemble_model`: Final ensemble model
- `scaler`: Feature scaler
- `selected_features`: List of feature names
- `feature_importance`: Feature importance scores
- `evaluation_results`: Model evaluation metrics
- `model_weights`: Weights for ensemble (optional)

## 🧪 Testing

### Manual Testing with cURL

```bash
# Health check
curl http://localhost:8000/health

# Price prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "lokasi": "Tembalang",
    "kamar_tidur": 3,
    "luas_bangunan": "120"
  }'
```

### Testing with Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "lokasi": "Tembalang",
    "kamar_tidur": 3,
    "kamar_mandi": 2,
    "luas_bangunan": "120",
    "luas_tanah": "150"
}

response = requests.post(url, json=data)
print(response.json())
```

## 📊 Model Performance

The model uses an ensemble of:
- **XGBoost**: Gradient boosting with regularization
- **Random Forest**: Ensemble decision trees
- **Gradient Boosting**: Sequential boosting algorithm

Model evaluation uses metrics:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

## 🚀 Deployment

### Docker Deployment (Recommended)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_DOWNLOAD_URL=your_model_url

EXPOSE 8000

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t smarthome-api .
docker run -p 8000:8000 smarthome-api
```

### Cloud Deployment

This API can be deployed to:
- **Heroku**: Add a `Procfile`
- **Google Cloud Run**: Build container image
- **AWS Elastic Beanstalk**: Package as Python application
- **Railway/Render**: Direct deployment from GitHub

## 📁 Project Structure

```
smarthome-valuator/
│
├── main.py                 # Main application file
├── requirements.txt        # Python dependencies
├── .env                   # Environment variables (not in repo)
├── .gitignore            # Git ignore file
├── README.md             # This file
│
├── models/               # Model files directory
│   └── smarthome_complete_pipeline.pkl
│
└── docs/                 # Additional documentation (optional)
    ├── API.md
    └── MODEL.md
```

## 🤝 Contributing

Contributions are always welcome! To contribute:

1. Fork this repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Create a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👥 Authors

- **Muhammad Rilly Ayidan** - *Initial work* - [YourGithub](https://github.com/rillyayidan)

## 🙏 Acknowledgments

- Semarang property dataset
- FastAPI framework
- Scikit-learn & XGBoost communities
- Contributors and testers

## 📞 Contact

Project Link: [https://github.com/rillyayidan/smarthome-valuator](https://github.com/rillyayidan/smarthome-valuator)

---

**⭐ If this project helps you, don't forget to give it a star!**
