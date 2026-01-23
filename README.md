# ML Service

A clean machine learning service with training and inference capabilities.

## Project Structure

```
ml-service/
├── ml_service/
│   ├── __init__.py
│   ├── model.py          # Model definition
│   ├── optimizer.py      # Hyperparameter optimization module
│   └── price_optimizer.py # Price optimization module
├── scripts/
│   ├── train.py          # General training script
│   ├── train_real_estate.py  # LightGBM training for real-estate data
│   └── optimize_price.py    # Price optimization script
├── api/
│   ├── __init__.py
│   └── main.py           # FastAPI service
├── models/               # Trained models (gitignored)
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

#### General Model Training

Train a model:
```bash
python scripts/train.py
```

#### Real-Estate Prediction Model (LightGBM)

Train a LightGBM model for predicting `sold_within_180d` from real-estate data:

```bash
python scripts/train_real_estate.py --data path/to/your/data.csv
```

**Options:**
- `--data`: Path to CSV file with real-estate data (required)
- `--categorical`: List of categorical feature names (auto-detected if not provided)
- `--test-size`: Test set size (default: 0.2)
- `--calibration-method`: Calibration method - 'isotonic' or 'sigmoid' (default: isotonic)
- `--output-dir`: Output directory for models (default: models/)

**Example:**
```bash
python scripts/train_real_estate.py \
  --data data/real_estate.csv \
  --categorical property_type neighborhood city \
  --calibration-method isotonic \
  --test-size 0.2
```

**Features:**
- Handles both categorical and numeric features automatically
- Probability calibration for better probability estimates
- Comprehensive evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
- Saves both base and calibrated models
- Saves feature information and metrics to JSON files

**Data Format:**
Your CSV file should contain:
- A column named `sold_within_180d` (binary target: 0 or 1)
- Feature columns (categorical and/or numeric)

### Price Optimization

Optimize listing price to maximize expected revenue (price × sale_probability):

```bash
python scripts/optimize_price.py \
  --model-dir models \
  --features property_features.csv \
  --base-price 500000
```

**Options:**
- `--model-dir`: Directory containing trained model (default: models/)
- `--features`: Path to CSV (single row) or JSON file with property features
- `--base-price`: Base/reference price (required)
- `--min-price`: Minimum price to test (default: 0.5 * base_price)
- `--max-price`: Maximum price to test (default: 1.5 * base_price)
- `--num-points`: Number of price points to test (default: 100)
- `--price-step`: Fixed step size between prices (overrides num_points)
- `--strategy`: 'linear' or 'log' spacing (default: linear)
- `--top-n`: Number of top results in leaderboard (default: 10)
- `--output`: Output CSV file for leaderboard (optional)

**Example:**
```bash
python scripts/optimize_price.py \
  --model-dir models \
  --features property.json \
  --base-price 500000 \
  --min-price 400000 \
  --max-price 700000 \
  --num-points 200 \
  --top-n 20 \
  --output price_leaderboard.csv
```

**Features:**
- Simulates multiple price points
- Calls prediction function for each price
- Maximizes expected revenue (price × sale_probability)
- Returns optimal price and top-N leaderboard
- Supports linear or logarithmic price spacing

**Python API Usage:**
```python
from ml_service.price_optimizer import PriceOptimizer

# Create prediction function
def predict_fn(features):
    # Your prediction logic
    return sale_probability

# Create optimizer
optimizer = PriceOptimizer(
    prediction_function=predict_fn,
    price_column='price'
)

# Optimize
best_price, leaderboard = optimizer.optimize(
    features={'bedrooms': 3, 'bathrooms': 2, ...},
    base_price=500000,
    num_points=100,
    top_n=10
)

print(f"Optimal price: ${best_price.price:,.2f}")
print(f"Expected revenue: ${best_price.expected_revenue:,.2f}")
```

### Running the API

Start the FastAPI service:
```bash
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /health` - Health check
- `POST /predict` - Predict sale probability for a property
- `POST /recommend_price` - Optimize price to maximize expected revenue
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)

### API Usage Examples

#### Predict Sale Probability

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "price": 500000,
      "bedrooms": 3,
      "bathrooms": 2,
      "square_feet": 1500,
      "property_type": "house",
      "neighborhood": "downtown",
      "city": "san_francisco"
    }
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "probability": 0.75,
  "probabilities": {
    "not_sold": 0.25,
    "sold_within_180d": 0.75
  }
}
```

#### Recommend Optimal Price

```bash
curl -X POST "http://localhost:8000/recommend_price" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "bedrooms": 3,
      "bathrooms": 2,
      "square_feet": 1500,
      "property_type": "house",
      "neighborhood": "downtown",
      "city": "san_francisco"
    },
    "base_price": 500000,
    "min_price": 400000,
    "max_price": 700000,
    "num_points": 200,
    "top_n": 10
  }'
```

**Response:**
```json
{
  "best_price": {
    "price": 525000.0,
    "sale_probability": 0.82,
    "expected_revenue": 430500.0,
    "rank": 1
  },
  "leaderboard": [
    {
      "price": 525000.0,
      "sale_probability": 0.82,
      "expected_revenue": 430500.0,
      "rank": 1
    },
    ...
  ]
}
```

### Request Models

#### PropertyFeatures
All endpoints accept a `PropertyFeatures` object with the following optional fields:
- `price` (float): Property price
- `bedrooms` (int): Number of bedrooms
- `bathrooms` (float): Number of bathrooms
- `square_feet` (float): Square footage
- `lot_size` (float): Lot size
- `year_built` (int): Year built
- `property_type` (string): Type of property
- `neighborhood` (string): Neighborhood
- `city` (string): City
- Additional fields are allowed (extra="allow")

