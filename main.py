from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
import pickle
import re
from datetime import datetime
import joblib
import uvicorn
import logging
from pathlib import Path
import requests
import os
from dotenv import load_dotenv

load_dotenv()

def download_model_if_needed():
    model_path = Path("./models/smarthome_complete_pipeline.pkl")
    if not model_path.exists():
        print("📥 Model not found. Downloading from remote URL...")
        url = os.getenv("MODEL_DOWNLOAD_URL")
        if not url:
            raise ValueError("MODEL_DOWNLOAD_URL is not set in .env")
        response = requests.get(url)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded and saved.")
    else:
        print("✅ Model file already exists. Skipping download.")


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SmartHome Price Predictor API",
    description="API untuk prediksi harga rumah di Semarang menggunakan machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
pipeline_components = None
predictor = None

class PropertyInput(BaseModel):
    """Input model untuk prediksi harga rumah"""
    lokasi: str = Field(..., description="Lokasi properti di Semarang", example="Tembalang")
    kamar_tidur: Optional[Union[int, str]] = Field(None, description="Jumlah kamar tidur", example=3)
    kamar_mandi: Optional[Union[int, str]] = Field(None, description="Jumlah kamar mandi", example=2)
    luas_tanah: Optional[Union[int, str]] = Field(None, description="Luas tanah dalam m² atau string", example="150")
    luas_bangunan: Optional[Union[int, str]] = Field(None, description="Luas bangunan dalam m² atau string", example="120")
    carport: Optional[Union[int, str]] = Field(None, description="Jumlah carport", example=1)
    daya_listrik: Optional[Union[int, str]] = Field(None, description="Daya listrik dalam VA", example=1300)
    jumlah_lantai: Optional[Union[int, str]] = Field(None, description="Jumlah lantai", example=1)
    kondisi_properti: Optional[str] = Field(None, description="Kondisi properti", example="Bagus")
    kondisi_perabotan: Optional[str] = Field(None, description="Kondisi perabotan", example="Tidak Berperabot")

    @validator('kamar_tidur', 'kamar_mandi', 'carport', 'jumlah_lantai', pre=True)
    def validate_integers(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            # Extract numbers from string
            numbers = re.findall(r'\d+', v)
            if numbers:
                return int(numbers[0])
            return None
        return v

    @validator('luas_tanah', 'luas_bangunan', 'daya_listrik', pre=True)
    def validate_numeric_with_units(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            # Remove common units and extract numbers
            clean_v = re.sub(r'[^\d.,]', '', v)
            clean_v = clean_v.replace(',', '.')
            try:
                return float(clean_v)
            except ValueError:
                return None
        return v

class PredictionResponse(BaseModel):
    """Response model untuk hasil prediksi"""
    predicted_price: float = Field(..., description="Harga prediksi dalam Rupiah")
    predicted_price_formatted: str = Field(..., description="Harga prediksi dalam format yang mudah dibaca")
    predicted_price_miliar: float = Field(..., description="Harga prediksi dalam miliar Rupiah")
    confidence_interval: Optional[str] = Field(None, description="Interval kepercayaan 95%")
    uncertainty: Optional[float] = Field(None, description="Tingkat ketidakpastian")
    property_category: str = Field(..., description="Kategori harga properti")
    zona: str = Field(..., description="Zona lokasi yang telah dimapping")
    preprocessing_info: Optional[Dict] = Field(None, description="Info preprocessing untuk debugging")

class BatchPredictionRequest(BaseModel):
    """Request model untuk prediksi batch"""
    properties: List[PropertyInput] = Field(..., description="List properti untuk diprediksi")

class HealthResponse(BaseModel):
    """Response model untuk health check"""
    status: str
    message: str
    timestamp: str
    models_loaded: bool

class DataProcessor:
    """Class untuk preprocessing data sesuai dengan pipeline training"""
    
    def __init__(self):
        self.zona_mapping = self._create_zona_mapping()
        self.default_values = self._get_default_values()
        
    def _create_zona_mapping(self):
        """Create mapping dari lokasi spesifik ke zona Semarang"""
        mapping = {
            # Semarang Barat
            "Semarang Barat": "Semarang Barat",
            "Kalibanteng": "Semarang Barat", "Kalibanteng Kulon": "Semarang Barat",
            "Kalibanteng kidul": "Semarang Barat", "Krapyak": "Semarang Barat",
            "Manyaran": "Semarang Barat", "Ngaliyan": "Semarang Barat",
            "Tugu": "Semarang Barat", "Tugurejo": "Semarang Barat",
            "Jerakah": "Semarang Barat", "Puspogiwang": "Semarang Barat",
            "Gajah Mungkur": "Semarang Barat", "Sampangan": "Semarang Barat",
            "Puspowarno": "Semarang Barat", "Puri Anjasmoro": "Semarang Barat",
            "Anjasmoro": "Semarang Barat", "Karangayu": "Semarang Barat",
            "Pusponjolo": "Semarang Barat", "Graha Padma": "Semarang Barat",
            "Kembang Arum": "Semarang Barat", "Tawangmas": "Semarang Barat",
            "Pamularsih": "Semarang Barat", "Mijen": "Semarang Barat",
            "Simongan": "Semarang Barat", "Bongsari": "Semarang Barat",
            "BSB City": "Semarang Barat",

            # Semarang Timur
            "Semarang Timur": "Semarang Timur",
            "Pedurungan": "Semarang Timur", "Tlogosari": "Semarang Timur",
            "Genuk": "Semarang Timur", "Gayamsari": "Semarang Timur",
            "Kedungmundu": "Semarang Timur", "Ketileng": "Semarang Timur",
            "Bangetayu": "Semarang Timur", "Bangetayu Wetan": "Semarang Timur",
            "Muktiharjo": "Semarang Timur", "Gemah": "Semarang Timur",
            "Plamongan": "Semarang Timur", "Meteseh": "Semarang Timur",
            "Mlatiharjo": "Semarang Timur", "Banyumanik": "Semarang Timur",
            "Tembalang": "Semarang Timur", "Bukit Sari": "Semarang Timur",
            "Sendangmulyo": "Semarang Timur", "Sambiroto": "Semarang Timur",
            "Srondol": "Semarang Timur", "Pudak Payung": "Semarang Timur",
            "Ngesrep": "Semarang Timur", "Jangli": "Semarang Timur",
            "Penggaron": "Semarang Timur", "Citragrand": "Semarang Timur",
            "Rejosari": "Semarang Timur", "Kalicari": "Semarang Timur",
            "Majapahit": "Semarang Timur", "Pedalangan": "Semarang Timur",
            "Kaligawe": "Semarang Timur",

            # Semarang Utara
            "Semarang Utara": "Semarang Utara",
            "Tanah Mas": "Semarang Utara", "Panggung": "Semarang Utara",
            "Kuningan": "Semarang Utara", "Plombokan": "Semarang Utara",
            "Tanjung Mas": "Semarang Utara", "Bandarharjo": "Semarang Utara",
            "Kampung Kali": "Semarang Utara",

            # Semarang Selatan
            "Semarang Selatan": "Semarang Selatan",
            "Wonodri": "Semarang Selatan", "Pleburan": "Semarang Selatan",
            "Candisari": "Semarang Selatan", "Jatingaleh": "Semarang Selatan",
            "Candi Golf": "Semarang Selatan", "Jomblang": "Semarang Selatan",
            "Lamper": "Semarang Selatan", "Karang Anyar": "Semarang Selatan",
            "Karang Rejo": "Semarang Selatan", "Karang Tempel": "Semarang Selatan",
            "Karang kidul": "Semarang Selatan", "Siranda": "Semarang Selatan",
            "Sompok": "Semarang Selatan", "Mugosari": "Semarang Selatan",
            "Tlaga Bodas": "Semarang Selatan", "Gajahmada": "Semarang Selatan",
            "Sultan Agung": "Semarang Selatan", "Peterongan": "Semarang Selatan",
            "Dr Cipto Mangunkusomo": "Semarang Selatan", "Atmodirono": "Semarang Selatan",
            "Bulustalan": "Semarang Selatan", "Pandansari": "Semarang Selatan",
            "Gajahmungkur": "Semarang Selatan", "Gunung Pati": "Semarang Selatan",
            "Barusari": "Semarang Selatan", "Pati Wetan": "Semarang Selatan",

            # Semarang Tengah
            "Semarang Tengah": "Semarang Tengah",
            "Simpang Lima": "Semarang Tengah", "Pemuda": "Semarang Tengah",
            "Sekayu": "Semarang Tengah", "Gabahan": "Semarang Tengah",
            "Kranggan": "Semarang Tengah", "Jagalan": "Semarang Tengah",
            "Miroto": "Semarang Tengah", "Pindrikan": "Semarang Tengah",
            "Pekunden": "Semarang Tengah", "Purwosari": "Semarang Tengah",
            "Pendirikan": "Semarang Tengah", "Brumbungan": "Semarang Tengah",
            "Mataram": "Semarang Tengah", "Kartini": "Semarang Tengah",
            "Bugangan": "Semarang Tengah", "Citarum": "Semarang Tengah",
            "Indraprasta": "Semarang Tengah", "Sidodadi Timur": "Semarang Tengah",
            "Kawi": "Semarang Tengah", "Halmahera": "Semarang Tengah",
            "Kaliwungu": "Semarang Tengah", "Krakatau": "Semarang Tengah",
            "Nias": "Semarang Tengah", "Semeru": "Semarang Tengah",
            "Sri Rejeki": "Semarang Tengah", "Kenconowungu": "Semarang Tengah",
            "Dempel": "Semarang Tengah", "Papandayan": "Semarang Tengah",
            "pekunden": "Semarang Tengah", "Cabean": "Semarang Tengah",
            "Gedung Batu": "Semarang Tengah", "Greenwood": "Semarang Tengah",
            "Karang Turi": "Semarang Tengah", "Kauman": "Semarang Tengah",
            "Purwodinatan": "Semarang Tengah", "Sarirejo": "Semarang Tengah",
            "papandayan": "Semarang Tengah"
        }

        # Lokasi luar kota Semarang (sekitar)
        luar_kota = [
            "Ungaran", "Ungaran Barat", "Ungaran Timur", "Bawen",
            "Bergas", "Boja", "Bandungan", "Mranggen", "Bringin",
            "Pringapus", "Sumowono", "Suruh", "Tengaran", "Tuntang",
            "Getasan", "Pabelan", "Banjardowo", "Kedung Pane",
            "Tengger", "Mangunsari"
        ]

        for lokasi in luar_kota:
            mapping[lokasi] = "Semarang Lainnya"

        return mapping
    
    def _get_default_values(self):
        """Get default values berdasarkan modus/median dari training data"""
        return {
            'kamar_tidur': 3,  # Modus
            'kamar_mandi': 2,  # Modus
            'luas_tanah': 150,  # Median
            'luas_bangunan': 120,  # Median
            'carport': 1,  # Modus
            'daya_listrik': 1300,  # Default berdasarkan kategori
            'jumlah_lantai': 1,  # Modus
            'kondisi_properti': 'Bagus',  # Modus
            'kondisi_perabotan': 'Tidak Berperabot'  # Modus
        }
    
    def clean_numeric_string(self, value):
        """Clean numeric string dan convert ke float"""
        if pd.isna(value) or value is None:
            return None
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove units and clean
            cleaned = re.sub(r'[^\d.,]', '', value)
            cleaned = cleaned.replace(',', '.')
            
            try:
                return float(cleaned)
            except ValueError:
                return None
        
        return None
    
    def map_lokasi_to_zona(self, lokasi):
        """Map lokasi ke zona dengan fuzzy matching"""
        if pd.isna(lokasi) or lokasi is None:
            return "Semarang Lainnya"

        # Clean lokasi string
        if isinstance(lokasi, str):
            lokasi = lokasi.strip()
            if ", Semarang" in lokasi:
                lokasi = lokasi.replace(", Semarang", "").strip()

        # Direct mapping
        if lokasi in self.zona_mapping:
            return self.zona_mapping[lokasi]

        # Fuzzy matching
        for key, zona in self.zona_mapping.items():
            if isinstance(key, str) and isinstance(lokasi, str):
                if key.lower() in lokasi.lower() or lokasi.lower() in key.lower():
                    return zona

        return "Semarang Lainnya"
    
    def classify_daya(self, daya):
        """Klasifikasi kategori daya listrik"""
        if pd.isna(daya) or daya is None:
            return 'Menengah_1300VA'
        
        if daya <= 450:
            return 'Dasar_450VA'
        elif daya <= 900:
            return 'Standar_900VA'
        elif daya <= 1300:
            return 'Menengah_1300VA'
        elif daya <= 2200:
            return 'Tinggi_2200VA'
        elif daya <= 3500:
            return 'Premium_3500VA'
        elif daya <= 5500:
            return 'Luxury_5500VA'
        else:
            return 'Ultra_5500VA_Plus'
    
    def get_daya_default_by_price_category(self, luas_bangunan, kamar_tidur):
        """Estimasi daya listrik berdasarkan luas bangunan dan kamar tidur"""
        if luas_bangunan >= 200 or kamar_tidur >= 4:
            return 2200
        elif luas_bangunan >= 120 or kamar_tidur >= 3:
            return 1300
        elif luas_bangunan >= 70:
            return 900
        else:
            return 450
    
    def get_lantai_default_by_size(self, luas_bangunan):
        """Estimasi jumlah lantai berdasarkan luas bangunan"""
        if luas_bangunan >= 150:
            return 2
        else:
            return 1
    
    def standardize_kondisi(self, kondisi, kondisi_type='properti'):
        """Standardize kondisi properti atau perabotan"""
        if pd.isna(kondisi) or kondisi is None:
            if kondisi_type == 'properti':
                return 'Bagus'
            else:
                return 'Tidak Berperabot'
        
        kondisi = str(kondisi).strip().title()
        
        if kondisi_type == 'properti':
            # Mapping untuk kondisi properti
            kondisi_mapping = {
                'Bagus': 'Bagus',
                'Baik': 'Bagus',
                'Good': 'Bagus',
                'Sangat Bagus': 'Bagus',
                'Siap Huni': 'Bagus',
                'Terawat': 'Bagus',
                'Renovasi': 'Renovasi',
                'Perlu Renovasi': 'Renovasi',
                'Rusak': 'Renovasi',
                'Butuh Renovasi': 'Renovasi'
            }
        else:
            # Mapping untuk kondisi perabotan
            kondisi_mapping = {
                'Tidak Berperabot': 'Tidak Berperabot',
                'Unfurnished': 'Tidak Berperabot',
                'Kosong': 'Tidak Berperabot',
                'Semi Furnished': 'Semi Furnished',
                'Semi Berperabot': 'Semi Furnished',
                'Sebagian Berperabot': 'Semi Furnished',
                'Furnished': 'Furnished',
                'Berperabot': 'Furnished',
                'Full Furnished': 'Furnished',
                'Lengkap': 'Furnished'
            }
        
        # Cari yang paling cocok
        for key, value in kondisi_mapping.items():
            if key.lower() in kondisi.lower() or kondisi.lower() in key.lower():
                return value
        
        # Default
        return kondisi_mapping[list(kondisi_mapping.keys())[0]]

def load_pipeline():
    global pipeline_components, predictor

    try:
        pipeline_path = Path("./models/smarthome_complete_pipeline.pkl")
        if not pipeline_path.exists():
            logger.error(f"❌ Model file not found at: {pipeline_path.resolve()}")
            return False

        logger.info(f"📦 Loading model from: {pipeline_path.resolve()}")
        with open(pipeline_path, "rb") as f:
            pipeline_components = pickle.load(f)

        predictor = SmartHomePricePrediction(pipeline_components)
        logger.info("✅ Pipeline loaded successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Error loading pipeline: {str(e)}")
        return False
class SmartHomePricePrediction:
    """Class untuk prediksi harga rumah dengan preprocessing lengkap"""
    
    def __init__(self, components):
        """Initialize dengan komponen pipeline"""
        self.models = components['models']
        self.ensemble_model = components['ensemble_model']
        self.scaler = components['scaler']
        self.selected_features = components['selected_features']
        self.feature_importance = components['feature_importance']
        self.evaluation_results = components['evaluation_results']
        self.model_weights = components.get('model_weights', [])
        
        # Initialize data processor
        self.data_processor = DataProcessor()
    
    def classify_harga(self, price):
        """Klasifikasi kategori harga"""
        if price < 500e6:  # < 500 juta
            return 'Ekonomis'
        elif price < 1000e6:  # 500 juta - 1 miliar
            return 'Menengah Bawah'
        elif price < 2000e6:  # 1 - 2 miliar
            return 'Menengah'
        elif price < 3500e6:  # 2 - 3.5 miliar
            return 'Menengah Atas'
        elif price < 6000e6:  # 3.5 - 6 miliar
            return 'Premium'
        else:  # > 6 miliar
            return 'Luxury'
    
    def preprocess_input(self, input_data: PropertyInput):
        """Preprocess input dengan imputation dan validation lengkap"""
        
        preprocessing_info = {}
        
        # 1. Extract dan clean data
        raw_data = {
            'Lokasi': input_data.lokasi,
            'Kamar Tidur': input_data.kamar_tidur,
            'Kamar Mandi': input_data.kamar_mandi,
            'Luas Tanah': input_data.luas_tanah,
            'Luas Bangunan': input_data.luas_bangunan,
            'Carport': input_data.carport,
            'Daya Listrik': input_data.daya_listrik,
            'Jumlah Lantai': input_data.jumlah_lantai,
            'Kondisi Properti': input_data.kondisi_properti,
            'Kondisi Perabotan': input_data.kondisi_perabotan
        }
        
        # 2. Clean numeric values
        numeric_fields = ['Kamar Tidur', 'Kamar Mandi', 'Luas Tanah', 'Luas Bangunan', 
                         'Carport', 'Daya Listrik', 'Jumlah Lantai']
        
        for field in numeric_fields:
            if field in ['Luas Tanah', 'Luas Bangunan', 'Daya Listrik']:
                raw_data[field] = self.data_processor.clean_numeric_string(raw_data[field])
            else:
                if raw_data[field] is not None:
                    try:
                        raw_data[field] = float(raw_data[field])
                    except (ValueError, TypeError):
                        raw_data[field] = None
        
        # 3. Handle missing values dengan imputation
        missing_fields = []
        
        # Kamar Tidur & Kamar Mandi
        if raw_data['Kamar Tidur'] is None or raw_data['Kamar Tidur'] <= 0:
            raw_data['Kamar Tidur'] = self.data_processor.default_values['kamar_tidur']
            missing_fields.append('Kamar Tidur')
        
        if raw_data['Kamar Mandi'] is None or raw_data['Kamar Mandi'] <= 0:
            raw_data['Kamar Mandi'] = self.data_processor.default_values['kamar_mandi']
            missing_fields.append('Kamar Mandi')
        
        # Luas Tanah & Luas Bangunan
        if raw_data['Luas Tanah'] is None or raw_data['Luas Tanah'] <= 0:
            raw_data['Luas Tanah'] = self.data_processor.default_values['luas_tanah']
            missing_fields.append('Luas Tanah')
        
        if raw_data['Luas Bangunan'] is None or raw_data['Luas Bangunan'] <= 0:
            raw_data['Luas Bangunan'] = self.data_processor.default_values['luas_bangunan']
            missing_fields.append('Luas Bangunan')
        
        # Carport
        if raw_data['Carport'] is None:
            raw_data['Carport'] = self.data_processor.default_values['carport']
            missing_fields.append('Carport')
        
        # Daya Listrik - estimasi berdasarkan size
        if raw_data['Daya Listrik'] is None or raw_data['Daya Listrik'] <= 0:
            raw_data['Daya Listrik'] = self.data_processor.get_daya_default_by_price_category(
                raw_data['Luas Bangunan'], raw_data['Kamar Tidur']
            )
            missing_fields.append('Daya Listrik')
        
        # Jumlah Lantai - estimasi berdasarkan luas
        if raw_data['Jumlah Lantai'] is None or raw_data['Jumlah Lantai'] <= 0:
            raw_data['Jumlah Lantai'] = self.data_processor.get_lantai_default_by_size(
                raw_data['Luas Bangunan']
            )
            missing_fields.append('Jumlah Lantai')
        
        # Kondisi Properti & Perabotan
        raw_data['Kondisi Properti'] = self.data_processor.standardize_kondisi(
            raw_data['Kondisi Properti'], 'properti'
        )
        raw_data['Kondisi Perabotan'] = self.data_processor.standardize_kondisi(
            raw_data['Kondisi Perabotan'], 'perabotan'
        )
        
        preprocessing_info['missing_fields'] = missing_fields
        preprocessing_info['imputed_values'] = {field: raw_data[field] for field in missing_fields}
        
        # 4. Validation ranges
        if raw_data['Kamar Tidur'] > 10:
            raw_data['Kamar Tidur'] = 10
        if raw_data['Kamar Mandi'] > 10:
            raw_data['Kamar Mandi'] = 10
        if raw_data['Luas Tanah'] > 2000:
            raw_data['Luas Tanah'] = 2000
        if raw_data['Luas Bangunan'] > 1000:
            raw_data['Luas Bangunan'] = 1000
        if raw_data['Carport'] > 5:
            raw_data['Carport'] = 5
        if raw_data['Jumlah Lantai'] > 4:
            raw_data['Jumlah Lantai'] = 4
        
        # Convert to DataFrame
        df = pd.DataFrame([raw_data])
        
        # 5. Process lokasi to zona
        df['Zona'] = df['Lokasi'].apply(self.data_processor.map_lokasi_to_zona)
        preprocessing_info['zona_mapped'] = df['Zona'].iloc[0]
        
        # One-hot encoding untuk zona
        all_zones = ['Semarang Barat', 'Semarang Timur', 'Semarang Utara', 
                    'Semarang Selatan', 'Semarang Tengah', 'Semarang Lainnya']
        
        for zone in all_zones:
            df[f'Zona_{zone}'] = (df['Zona'] == zone).astype(int)
        
        df = df.drop(columns=['Lokasi', 'Zona'])
        
        # 6. Process daya listrik
        df['Kategori_Daya'] = df['Daya Listrik'].apply(self.data_processor.classify_daya)
        
        # One-hot encoding untuk daya
        all_daya = ['Dasar_450VA', 'Standar_900VA', 'Menengah_1300VA', 'Tinggi_2200VA',
                   'Premium_3500VA', 'Luxury_5500VA', 'Ultra_5500VA_Plus']
        
        for daya in all_daya:
            df[f'Daya_{daya}'] = (df['Kategori_Daya'] == daya).astype(int)
        
        df = df.drop(columns=['Kategori_Daya'])
        
        # 7. Process kondisi properti dan perabotan
        # Kondisi Properti
        all_kondisi_properti = ['Bagus', 'Renovasi']
        for kondisi in all_kondisi_properti:
            df[f'Kondisi_Properti_{kondisi}'] = (df['Kondisi Properti'] == kondisi).astype(int)
        
        # Kondisi Perabotan
        all_kondisi_perabotan = ['Tidak Berperabot', 'Semi Furnished', 'Furnished']
        for kondisi in all_kondisi_perabotan:
            df[f'Kondisi_Perabotan_{kondisi}'] = (df['Kondisi Perabotan'] == kondisi).astype(int)
        
        df = df.drop(columns=['Kondisi Properti', 'Kondisi Perabotan'])
        
# 8. Create derived features
        df['Rasio_Bangunan_Tanah'] = df['Luas Bangunan'] / df['Luas Tanah']
        df['Total_Luas'] = df['Luas Bangunan'] + df['Luas Tanah']
        df['Luas_per_Kamar'] = df['Luas Bangunan'] / df['Kamar Tidur']
        df['Rasio_Kamar_Mandi'] = df['Kamar Mandi'] / df['Kamar Tidur']
        df['Daya_per_Luas'] = df['Daya Listrik'] / df['Luas Bangunan']
        
        # Additional features that might be expected by the model
        df['Price_per_sqm'] = 0  # Will be calculated after prediction if needed
        df['Kamar_x_Daya'] = df['Kamar Tidur'] * df['Daya Listrik']
        df['Luas Tanah_log'] = np.log1p(df['Luas Tanah'])
        df['Luas Bangunan_log'] = np.log1p(df['Luas Bangunan'])
        
        # More potential derived features
        df['Total_Rooms'] = df['Kamar Tidur'] + df['Kamar Mandi']
        df['Luxury_Score'] = (df['Daya Listrik'] / 1000) * df['Luas Bangunan'] * df['Jumlah Lantai']
        df['Efficiency_Score'] = df['Luas Bangunan'] / df['Luas Tanah']
        
        # 9. Select features yang digunakan oleh model
        # Handle case where some expected features might not be in selected_features
        try:
            df_selected = df[self.selected_features]
        except KeyError as e:
            # Log missing features and create them with default values
            missing_features = [col for col in self.selected_features if col not in df.columns]
            logger.warning(f"Missing features: {missing_features}")
            
            # Create missing features with default values
            for feature in missing_features:
                if 'log' in feature.lower():
                    df[feature] = 0
                elif 'ratio' in feature.lower() or 'per' in feature.lower():
                    df[feature] = 1
                elif 'score' in feature.lower():
                    df[feature] = df['Luas Bangunan'] * 0.1  # Some default score
                else:
                    df[feature] = 0
            
            # Try again to select features
            df_selected = df[self.selected_features]
        
        # 10. Scale features
        df_scaled = pd.DataFrame(
            self.scaler.transform(df_selected),
            columns=self.selected_features
        )
        
        preprocessing_info['features_shape'] = df_scaled.shape
        
        return df_scaled, preprocessing_info
    
    def predict(self, input_data: PropertyInput):
        """Prediksi harga dengan ensemble model"""
        try:
            # Preprocess input
            X_processed, preprocessing_info = self.preprocess_input(input_data)
            
            # Get predictions from all models
            predictions = []
            for model_name, model in self.models.items():
                pred = model.predict(X_processed)[0]
                predictions.append(pred)
            
            # Ensemble prediction
            if self.model_weights:
                ensemble_pred = np.average(predictions, weights=self.model_weights)
            else:
                ensemble_pred = np.mean(predictions)
            
            # Get final prediction from ensemble model if available
            if self.ensemble_model:
                final_pred = self.ensemble_model.predict(X_processed)[0]
            else:
                final_pred = ensemble_pred
            
            # Ensure positive price
            final_pred = max(final_pred, 0)
            
            # Format price
            price_formatted = f"Rp {final_pred:,.0f}".replace(',', '.')
            price_miliar = final_pred / 1e9
            
            # Get property category
            category = self.classify_harga(final_pred)
            
            # Get zona
            zona = preprocessing_info.get('zona_mapped', 'Unknown')
            
            # Calculate uncertainty (standard deviation of predictions)
            uncertainty = np.std(predictions) if len(predictions) > 1 else 0
            
            # Confidence interval (95%)
            if uncertainty > 0:
                ci_lower = final_pred - 1.96 * uncertainty
                ci_upper = final_pred + 1.96 * uncertainty
                confidence_interval = f"Rp {ci_lower:,.0f} - Rp {ci_upper:,.0f}".replace(',', '.')
            else:
                confidence_interval = None
            
            return {
                'predicted_price': final_pred,
                'predicted_price_formatted': price_formatted,
                'predicted_price_miliar': price_miliar,
                'confidence_interval': confidence_interval,
                'uncertainty': uncertainty,
                'property_category': category,
                'zona': zona,
                'preprocessing_info': preprocessing_info
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting SmartHome Price Predictor API...")
    success = load_pipeline()
    if not success:
        logger.error("Failed to load models. API will not function properly.")
    else:
        logger.info("API started successfully!")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
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

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if pipeline_components is not None else "unhealthy",
        message="API is running" if pipeline_components is not None else "Models not loaded",
        timestamp=datetime.now().isoformat(),
        models_loaded=pipeline_components is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(input_data: PropertyInput):
    """Prediksi harga rumah tunggal"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please check server logs.")
    
    try:
        logger.info(f"Predicting price for location: {input_data.lokasi}")
        result = predictor.predict(input_data)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(request: BatchPredictionRequest):
    """Prediksi harga untuk multiple properti"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please check server logs.")
    
    if len(request.properties) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 properties per batch request")
    
    try:
        results = []
        for i, property_data in enumerate(request.properties):
            try:
                result = predictor.predict(property_data)
                result['index'] = i
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to predict property {i}: {str(e)}")
                results.append({
                    'index': i,
                    'error': str(e),
                    'predicted_price': None
                })
        
        return {
            "total_properties": len(request.properties),
            "successful_predictions": len([r for r in results if 'error' not in r]),
            "failed_predictions": len([r for r in results if 'error' in r]),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    """Get information about loaded models"""
    if pipeline_components is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        return {
            "models": list(pipeline_components['models'].keys()),
            "selected_features": pipeline_components['selected_features'][:10],  # First 10 features
            "total_features": len(pipeline_components['selected_features']),
            "evaluation_results": pipeline_components.get('evaluation_results', {}),
            "feature_importance": dict(list(pipeline_components.get('feature_importance', {}).items())[:10])  # Top 10 features
        }
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/zones")
async def get_available_zones():
    """Get list of available zones/locations"""
    processor = DataProcessor()
    
    zones = {}
    for location, zone in processor.zona_mapping.items():
        if zone not in zones:
            zones[zone] = []
        zones[zone].append(location)
    
    return {
        "total_zones": len(zones),
        "zones": zones,
        "zone_names": list(zones.keys())
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "message": "Please check the API documentation at /docs"}

@app.exception_handler(422)
async def validation_error_handler(request, exc):
    return {"error": "Validation error", "details": exc.detail}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return {"error": "Internal server error", "message": "Please try again later"}

if __name__ == "__main__":
    # Configuration
    HOST = "0.0.0.0"
    PORT = 8000

    # Logging
    logger.info(f"Starting server on {HOST}:{PORT}")

    # ✅ Download model terlebih dahulu
    download_model_if_needed()

    # ✅ Load pipeline
    load_pipeline()

    # Run with uvicorn
    uvicorn.run(
        "main:app",  # Ubah sesuai nama file jika bukan main.py
        host=HOST,
        port=PORT,
        reload=False,  # Set True saat local dev
        log_level="info"
    )