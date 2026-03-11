# Kolommen Vertaling - Training Data

## Belangrijk: Technische Kolomnamen

De trainingsdata (`listings.csv`) gebruikt **Engelse technische kolomnamen** omdat:
1. Het model intern deze namen gebruikt
2. Code compatibiliteit behouden
3. Standaard data science conventies

**Voor presentaties/demo's:** Gebruik de Nederlandse vertaling hieronder.

## Kolom Mapping (Engels → Nederlands)

| Engels (CSV) | Nederlands (Presentatie) | Beschrijving |
|--------------|-------------------------|--------------|
| `asset_type` | **Asset Type** | Type vastgoed (logistics/office/resi) |
| `city` | **Stad** | Locatie van het property |
| `size_m2` | **Oppervlakte (m²)** | Totale oppervlakte in vierkante meters |
| `quality_score` | **Kwaliteitsscore** | Kwaliteit van 0-1 (0=laag, 1=hoog) |
| `noi_annual` | **NOI Jaarlijks** | Net Operating Income per jaar (€) |
| `cap_rate_market` | **Cap Rate Markt** | Markt cap rate (decimaal, bijv. 0.065 = 6.5%) |
| `interest_rate` | **Rentestand** | Huidige rentestand (decimaal, bijv. 0.025 = 2.5%) |
| `liquidity_index` | **Liquiditeitsindex** | Marktliquiditeit 0-1 (0=illiquide, 1=zeer liquide) |
| `list_price` | **Vraagprijs** | Originele vraagprijs bij listing (€) |
| `comp_median_price` | **Mediaan Prijs Vergelijkbaar** | Mediaan prijs vergelijkbare properties (€) |
| `sold_within_180d` | **Verkocht Binnen 180 Dagen** | Target: 1=ja, 0=nee |

## Asset Type Waarden

| Engels | Nederlands |
|--------|------------|
| `logistics` | Logistiek |
| `office` | Kantoor |
| `resi` | Residentieel |
| `retail` | Retail |
| `mixed` | Gemengd |

## Voorbeeld Data Rij

**Engels (CSV):**
```
logistics;Rotterdam;12000;0.82;620000;0.065;0.025;0.71;9500000;9900000;1
```

**Nederlandse Vertaling:**
- **Asset Type:** Logistiek
- **Stad:** Rotterdam
- **Oppervlakte:** 12.000 m²
- **Kwaliteitsscore:** 0.82 (82%)
- **NOI Jaarlijks:** €620.000
- **Cap Rate Markt:** 6.5%
- **Rentestand:** 2.5%
- **Liquiditeitsindex:** 0.71 (71%)
- **Vraagprijs:** €9.500.000
- **Mediaan Prijs Vergelijkbaar:** €9.900.000
- **Verkocht Binnen 180 Dagen:** Ja (1)

## Training vs Presentatie

### Voor Training (Code)
- Gebruik `listings.csv` met Engelse kolomnamen
- Model verwacht exact deze kolomnamen
- Script: `python scripts/train_real_estate.py --data data/listings.csv`

### Voor Presentatie/Demo
- Gebruik Nederlandse vertalingen hierboven
- Toon Nederlandse labels in UI
- Intern gebruikt code nog steeds Engelse namen

## Data Dictionary

Zie `DATA_DICTIONARY.md` voor volledige beschrijving van elke kolom.

## Waarom Engels in CSV?

1. **Code Compatibiliteit:** Python code gebruikt Engelse namen
2. **Data Science Standaard:** Meeste datasets gebruiken Engels
3. **Model Training:** Model verwacht specifieke kolomnamen
4. **API Intern:** API gebruikt Engelse feature namen intern

**Maar:** Alle API responses zijn in het Nederlands voor gebruikers!




