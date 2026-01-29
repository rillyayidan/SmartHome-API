# 🏠 SmartHome Valuator API

API backend untuk prediksi harga rumah di Semarang menggunakan Machine Learning. Dibangun dengan FastAPI dan menggunakan ensemble model untuk menghasilkan prediksi yang akurat.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Deskripsi

SmartHome Valuator adalah REST API yang memprediksi harga properti di Semarang berdasarkan berbagai fitur seperti lokasi, luas bangunan, jumlah kamar, dan fasilitas lainnya. API ini menggunakan ensemble model machine learning yang telah dilatih dengan data properti riil untuk memberikan estimasi harga yang akurat.

### ✨ Fitur Utama

- 🎯 **Prediksi Akurat**: Menggunakan ensemble model (XGBoost, Random Forest, Gradient Boosting)
- 🗺️ **Zone Mapping**: Mapping otomatis lokasi spesifik ke zona Semarang
- 🔄 **Smart Imputation**: Handling missing data dengan intelligent defaults
- 📊 **Batch Prediction**: Prediksi multiple properti sekaligus
- 🏷️ **Kategorisasi**: Klasifikasi properti dari Ekonomis hingga Luxury
- 📈 **Confidence Interval**: Estimasi ketidakpastian prediksi
- 🚀 **Auto Model Download**: Download model otomatis dari remote URL
- 📝 **API Documentation**: Interactive docs dengan Swagger UI

## 🛠️ Tech Stack

- **Framework**: FastAPI
- **ML Models**: XGBoost, Random Forest, Gradient Boosting
- **Data Processing**: Pandas, NumPy
- **Model Serialization**: Pickle, Joblib
- **Server**: Uvicorn
- **Validation**: Pydantic

## 📦 Instalasi

### Prerequisites

- Python 3.8 atau lebih tinggi
- pip (Python package manager)
- Virtual environment (recommended)

### Setup Lokal

1. **Clone repository**
```bash
git clone https://github.com/yourusername/smarthome-valuator.git
cd smarthome-valuator
```

2. **Buat virtual environment**
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

Buat file `.env` di root directory:
```env
MODEL_DOWNLOAD_URL=https://your-model-storage-url/smarthome_complete_pipeline.pkl
```

5. **Jalankan aplikasi**
```bash
python main.py
```

API akan berjalan di `http://localhost:8000`

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
Menampilkan informasi dasar API dan daftar endpoint yang tersedia.

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
Memeriksa status kesehatan API dan model.

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running",
  "timestamp": "2026-01-29T10:30:00",
  "models_loaded": true
}
```

#### 3. Prediksi Harga Tunggal
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

#### 4. Prediksi Batch
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
Menampilkan informasi tentang model yang sedang digunakan.

#### 6. Available Zones
```http
GET /zones
```
Menampilkan daftar zona dan lokasi yang tersedia.

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
| `lokasi` | string | ✅ | Lokasi properti di Semarang | "Tembalang" |
| `kamar_tidur` | int/string | ❌ | Jumlah kamar tidur | 3 |
| `kamar_mandi` | int/string | ❌ | Jumlah kamar mandi | 2 |
| `luas_tanah` | int/string | ❌ | Luas tanah (m²) | "150" |
| `luas_bangunan` | int/string | ❌ | Luas bangunan (m²) | "120" |
| `carport` | int/string | ❌ | Jumlah carport | 1 |
| `daya_listrik` | int/string | ❌ | Daya listrik (VA) | 1300 |
| `jumlah_lantai` | int/string | ❌ | Jumlah lantai | 2 |
| `kondisi_properti` | string | ❌ | Kondisi properti | "Bagus" |
| `kondisi_perabotan` | string | ❌ | Kondisi perabotan | "Furnished" |

### Notes:
- ✅ = Required field
- ❌ = Optional field (akan di-impute dengan nilai default)
- Semua field numeric dapat dikirim sebagai string dengan unit (akan dibersihkan otomatis)

## 🏷️ Kategori Properti

API mengklasifikasikan properti ke dalam 6 kategori:

| Kategori | Range Harga |
|----------|-------------|
| Ekonomis | < Rp 500 juta |
| Menengah Bawah | Rp 500 juta - Rp 1 miliar |
| Menengah | Rp 1 miliar - Rp 2 miliar |
| Menengah Atas | Rp 2 miliar - Rp 3.5 miliar |
| Premium | Rp 3.5 miliar - Rp 6 miliar |
| Luxury | > Rp 6 miliar |

## 🗺️ Zona Semarang

API mengenali 6 zona utama di Semarang:

1. **Semarang Barat**: Ngaliyan, Kalibanteng, Tugu, BSB City, dll.
2. **Semarang Timur**: Tembalang, Banyumanik, Pedurungan, Tlogosari, dll.
3. **Semarang Utara**: Tanah Mas, Panggung, Kuningan, dll.
4. **Semarang Selatan**: Wonodri, Candisari, Pleburan, dll.
5. **Semarang Tengah**: Simpang Lima, Pemuda, Sekayu, dll.
6. **Semarang Lainnya**: Ungaran, Bawen, Bergas, dll.

## 🔧 Konfigurasi

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

Model file harus berisi dictionary dengan keys:
- `models`: Dictionary of ML models
- `ensemble_model`: Final ensemble model
- `scaler`: Feature scaler
- `selected_features`: List of feature names
- `feature_importance`: Feature importance scores
- `evaluation_results`: Model evaluation metrics
- `model_weights`: Weights for ensemble (optional)

## 🧪 Testing

### Manual Testing dengan cURL

```bash
# Health check
curl http://localhost:8000/health

# Prediksi harga
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "lokasi": "Tembalang",
    "kamar_tidur": 3,
    "luas_bangunan": "120"
  }'
```

### Testing dengan Python

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

Model menggunakan ensemble dari:
- **XGBoost**: Gradient boosting dengan regularization
- **Random Forest**: Ensemble decision trees
- **Gradient Boosting**: Sequential boosting algorithm

Evaluasi model menggunakan metrics:
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

Build dan run:
```bash
docker build -t smarthome-api .
docker run -p 8000:8000 smarthome-api
```

### Cloud Deployment

API ini dapat di-deploy ke:
- **Heroku**: Tambahkan `Procfile`
- **Google Cloud Run**: Build container image
- **AWS Elastic Beanstalk**: Package sebagai Python application
- **Railway/Render**: Direct deployment dari GitHub

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

Kontribusi selalu welcome! Untuk berkontribusi:

1. Fork repository ini
2. Buat branch feature (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👥 Authors

- **Your Name** - *Initial work* - [YourGithub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Dataset properti Semarang
- FastAPI framework
- Scikit-learn & XGBoost communities
- Contributors dan testers

## 📞 Contact

Project Link: [https://github.com/rillyayidan/smarthome-valuator](https://github.com/rillyayidan/smarthome-valuator)

---

**⭐ Jika project ini membantu, jangan lupa untuk memberikan star!**
