# AI Model Training Guide

## Hoe train je het AI model?

### Stap 1: Bereid je data voor

Je hebt een CSV bestand nodig met alle property data. Het bestand moet deze kolommen bevatten:

**Verplichte kolommen:**
- `sold_within_180d` (0 of 1) - **Dit is je target variabele**
- `asset_type` (categorical: logistics, office, resi, etc.)
- `city` (categorical: Rotterdam, Amsterdam, etc.)
- `size_m2` (numeric)
- `quality_score` (numeric, 0-1)
- `noi_annual` (numeric)
- `cap_rate_market` (numeric, bijv. 0.065)
- `interest_rate` (numeric, bijv. 0.025)
- `liquidity_index` (numeric, 0-1)
- `list_price` (numeric)
- `comp_median_price` (numeric)

**Optionele kolommen (voor toekomstige uitbreiding):**
- `year_built`
- `occupancy_rate`
- `lease_term_remaining`
- `location_score`
- etc.

### Stap 2: Data kwaliteit

**Belangrijk:**
- ✅ Minimaal 50-100 voorbeelden (meer is beter!)
- ✅ Gebalanceerde data: ongeveer 50% verkocht (1), 50% niet verkocht (0)
- ✅ Geen missing values in belangrijke kolommen
- ✅ Realistische waarden (geen outliers die niet kloppen)
- ✅ Decimale getallen gebruiken punten, niet komma's (0.065 niet 0,065)

**Voorbeeld data structuur:**
```csv
asset_type,city,size_m2,quality_score,noi_annual,cap_rate_market,interest_rate,liquidity_index,list_price,comp_median_price,sold_within_180d
logistics,Rotterdam,12000,0.82,620000,0.065,0.025,0.71,9500000,9900000,1
office,Amsterdam,5000,0.60,310000,0.055,0.030,0.55,7200000,6900000,0
resi,Utrecht,110,0.75,42000,0.040,0.028,0.66,510000,495000,1
```

### Stap 3: Train het model

**Basis commando:**
```bash
cd ~/Desktop/ml-service
source .venv/bin/activate
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH

python scripts/train_real_estate.py --data data/your_data.csv
```

**Met opties:**
```bash
python scripts/train_real_estate.py \
  --data data/your_data.csv \
  --categorical asset_type city \
  --test-size 0.2 \
  --calibration-method isotonic \
  --output-dir models
```

**Parameters:**
- `--data`: Pad naar je CSV bestand
- `--categorical`: Lijst van categorische features (optioneel, wordt auto-gedetecteerd)
- `--test-size`: Percentage voor test set (default: 0.2 = 20%)
- `--calibration-method`: 'isotonic' of 'sigmoid' (default: isotonic)
- `--output-dir`: Waar model wordt opgeslagen (default: models/)

### Stap 4: Evalueer het model

Na training zie je:
- **Accuracy**: Hoe vaak het model correct voorspelt
- **Precision**: Van alle "verkocht" voorspellingen, hoeveel waren correct
- **Recall**: Van alle echte verkopen, hoeveel heeft het model gevonden
- **F1 Score**: Balans tussen precision en recall
- **ROC-AUC**: Hoe goed het model onderscheid maakt tussen verkocht/niet verkocht

**Goede scores:**
- Accuracy > 70% (voor real estate is dit goed)
- ROC-AUC > 0.75 (hoe hoger, hoe beter)
- F1 Score > 0.70

### Stap 5: Deploy het nieuwe model

Na training worden bestanden opgeslagen in `models/`:
- `lightgbm_calibrated.joblib` - Het getrainde model
- `feature_info.json` - Feature informatie
- `metrics.json` - Evaluatie metrics

**Om te deployen:**
1. Commit de nieuwe model bestanden:
   ```bash
   git add models/
   git commit -m "Update model with new training data"
   git push
   ```
2. Render zal automatisch redeployen
3. Je API gebruikt nu het nieuwe model

## Data verzameling tips

### Wat voor data heb je nodig?

**Minimaal:**
- 50-100 properties met bekende verkoopstatus
- Alle features die je nu gebruikt

**Idealiter:**
- 200+ properties
- Diverse asset types (logistics, office, resi)
- Verschillende steden
- Verschillende prijsklassen
- Verschillende marktomstandigheden (verschillende rentestanden)

### Waar haal je data vandaan?

1. **Eigen transacties**: Je eigen verkochte properties
2. **Public data**: Openbare transactie databases
3. **Partners**: Makelaars, brokers die data willen delen
4. **Data providers**: Commercial real estate data platforms
5. **Historische data**: Oude listings en hun verkoopstatus

### Data kwaliteit checklist

- [ ] Alle verplichte kolommen aanwezig
- [ ] Geen missing values in belangrijke features
- [ ] Realistische waarden (geen 0 of extreme outliers)
- [ ] Decimale getallen gebruiken punten (0.065)
- [ ] Categorische waarden consistent (Rotterdam, niet rotterdam of ROTTERDAM)
- [ ] Target variabele is 0 of 1 (niet "yes"/"no")
- [ ] Minimaal 50 voorbeelden, idealiter 200+

## Voorbeeld workflow

```bash
# 1. Bereid je data voor
# Zorg dat listings.csv alle data heeft

# 2. Clean de data (decimale komma's naar punten)
python - << 'PY'
import pandas as pd
df = pd.read_csv("data/listings.csv", sep=";")
for col in ["cap_rate_market", "interest_rate"]:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
df.to_csv("data/listings_clean.csv", index=False)
PY

# 3. Train het model
python scripts/train_real_estate.py --data data/listings_clean.csv

# 4. Check de metrics
cat models/metrics.json

# 5. Als tevreden, commit en push
git add models/
git commit -m "Retrained model with X new examples"
git push
```

## Troubleshooting

**"Not enough data" errors:**
- Je hebt minimaal 10-20 voorbeelden nodig
- Voor betere resultaten: 50+ voorbeelden

**"Model performance is poor":**
- Check of je data kwaliteit goed is
- Zorg voor meer diverse voorbeelden
- Check of features relevant zijn (correlatie met target)

**"Categorical features not working":**
- Zorg dat categorische waarden consistent zijn
- Gebruik `--categorical` flag om expliciet te specificeren

## Volgende stappen

1. **Meer data verzamelen**: Hoe meer, hoe beter het model
2. **Features toevoegen**: Nieuwe relevante features (bijv. occupancy_rate)
3. **Hyperparameter tuning**: Automatisch gedaan, maar je kunt handmatig aanpassen
4. **Model monitoring**: Track hoe goed het model presteert in productie

## Vragen?

- Check de training logs voor errors
- Bekijk `models/metrics.json` voor performance
- Test het model lokaal voordat je deployt







