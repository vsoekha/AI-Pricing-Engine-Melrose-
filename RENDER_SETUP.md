# Render Deployment Guide

## Step-by-Step Instructions

### 1. Create New Web Service in Render

1. Go to https://dashboard.render.com
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub account if not already connected
4. Select repository: **`AI-Pricing-Engine-Melrose-`**

### 2. Configure Service Settings

**Basic Settings:**
- **Name:** `ai-pricing-engine-melrose` (or any name you prefer)
- **Region:** Choose closest to your users (e.g., `Oregon (US West)`)
- **Branch:** `main`
- **Root Directory:** Leave **EMPTY** (or use `.`)

**Environment:**
- **Environment:** `Docker` (This is important - use Docker!)

**OR if not using Docker:**

**Build & Start:**
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

**Advanced Settings:**
- **Auto-Deploy:** `Yes` (automatically deploys on git push)
- **Health Check Path:** `/health`

### 3. Environment Variables

**No environment variables needed!** The app loads everything from the `models/` directory.

### 4. Plan Selection

- Choose **Free** plan (or paid if you need more resources)
- Free plan includes:
  - 750 hours/month
  - Spins down after 15 minutes of inactivity
  - First request after spin-down takes ~30 seconds

### 5. Create Service

Click **"Create Web Service"**

### 6. Wait for Deployment

- Render will:
  1. Clone your repository
  2. Build the Docker image (or install dependencies)
  3. Start the service
- This takes **3-5 minutes** for the first deployment
- Watch the logs to see progress

### 7. Verify Deployment

Once deployed, test these URLs:

1. **Root:** `https://your-service-name.onrender.com/`
   - Should show API info

2. **Health:** `https://your-service-name.onrender.com/health`
   - Should show: `{"status": "healthy", "model_loaded": true}`

3. **Docs:** `https://your-service-name.onrender.com/docs`
   - Interactive API documentation

4. **Recommend Price (GET):** `https://your-service-name.onrender.com/recommend_price`
   - Shows usage instructions

### 8. Test POST Endpoint

Use the `/docs` page or curl:

```bash
curl -X POST "https://your-service-name.onrender.com/recommend_price" \
  -H "Content-Type: application/json" \
  -d '{
    "asset_type": "logistics",
    "city": "Rotterdam",
    "size_m2": 12000,
    "quality_score": 0.82,
    "noi_annual": 620000,
    "cap_rate_market": 0.065,
    "interest_rate": 0.025,
    "liquidity_index": 0.71,
    "list_price": 9500000,
    "comp_median_price": 9900000
  }'
```

## Important Notes

### Using Docker (Recommended)
- ✅ Dockerfile is already configured
- ✅ Includes `libgomp1` for LightGBM
- ✅ Model files are in the repo
- ✅ Just select "Docker" as environment

### If Docker Doesn't Work
- Use "Python 3" environment
- Set build command: `pip install -r requirements.txt`
- Set start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
- **Note:** You may need to add system dependencies for LightGBM

### Troubleshooting

**If model doesn't load:**
1. Check `/health` endpoint for error details
2. Verify model files are in the repo (check GitHub)
3. Check Render logs for errors

**If you get "Method Not Allowed":**
- Make sure you're using POST, not GET
- Use `/docs` page to test properly

**If service is slow:**
- Free tier spins down after inactivity
- First request after spin-down takes ~30 seconds
- Consider upgrading to paid plan for always-on

## Frontend Integration

### JavaScript/Fetch Example

```javascript
const response = await fetch('https://your-service-name.onrender.com/recommend_price', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    asset_type: "logistics",
    city: "Rotterdam",
    size_m2: 12000,
    quality_score: 0.82,
    noi_annual: 620000,
    cap_rate_market: 0.065,
    interest_rate: 0.025,
    liquidity_index: 0.71,
    list_price: 9500000,
    comp_median_price: 9900000
  })
});

const result = await response.json();
console.log(result);
// {
//   "base_price": 9500000.0,
//   "base_sale_probability": 0.6667,
//   "recommended_price": 10925000.0,
//   "recommended_sale_probability": 0.6667,
//   "expected_uplift": 950000.0
// }
```

## Next Steps

1. Deploy on Render using steps above
2. Test all endpoints
3. Get your service URL
4. Integrate with your frontend (Base44 or any other)
5. If Base44 doesn't work, consider:
   - React + Vite
   - Next.js
   - Vue.js
   - Plain HTML + JavaScript









