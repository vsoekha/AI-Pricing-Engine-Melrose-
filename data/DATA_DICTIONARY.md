# Data Dictionary - Property Data Template

## Verplichte Kolommen

### asset_type (Categorical)
- **Type**: Text
- **Waarden**: `logistics`, `office`, `resi`, `retail`, `mixed`
- **Voorbeeld**: `logistics`
- **Beschrijving**: Type vastgoed asset

### city (Categorical)
- **Type**: Text
- **Waarden**: Stad naam (bijv. `Rotterdam`, `Amsterdam`, `Utrecht`)
- **Voorbeeld**: `Rotterdam`
- **Beschrijving**: Locatie van het property

### size_m2 (Numeric)
- **Type**: Number
- **Eenheid**: Vierkante meters
- **Voorbeeld**: `12000`
- **Beschrijving**: Totale oppervlakte van het property

### quality_score (Numeric)
- **Type**: Number (0-1)
- **Voorbeeld**: `0.82`
- **Beschrijving**: Kwaliteitsscore van het property (0 = laag, 1 = hoog)
- **Bepaling**: Op basis van conditie, locatie, features

### noi_annual (Numeric)
- **Type**: Number (Euro)
- **Voorbeeld**: `620000`
- **Beschrijving**: Net Operating Income per jaar (inkomsten - operationele kosten)

### cap_rate_market (Numeric)
- **Type**: Number (decimaal)
- **Voorbeeld**: `0.065` (betekent 6.5%)
- **Belangrijk**: Gebruik punt, niet komma (0.065 niet 0,065)
- **Beschrijving**: Markt cap rate (NOI / Property Value)

### interest_rate (Numeric)
- **Type**: Number (decimaal)
- **Voorbeeld**: `0.025` (betekent 2.5%)
- **Belangrijk**: Gebruik punt, niet komma
- **Beschrijving**: Huidige rentestand op moment van listing

### liquidity_index (Numeric)
- **Type**: Number (0-1)
- **Voorbeeld**: `0.71`
- **Beschrijving**: Marktliquiditeit (0 = illiquide, 1 = zeer liquide)
- **Bepaling**: Op basis van vraag/aanbod, transactievolume

### list_price (Numeric)
- **Type**: Number (Euro)
- **Voorbeeld**: `9500000`
- **Beschrijving**: Originele vraagprijs bij listing

### comp_median_price (Numeric)
- **Type**: Number (Euro)
- **Voorbeeld**: `9900000`
- **Beschrijving**: Mediaan prijs van vergelijkbare properties in de markt

### sold_within_180d (Binary - BELANGRIJK!)
- **Type**: Integer (0 of 1)
- **Waarden**: 
  - `1` = Verkocht binnen 180 dagen
  - `0` = Niet verkocht binnen 180 dagen
- **Voorbeeld**: `1`
- **Beschrijving**: **Dit is de target variabele voor het AI model**

## Optionele Kolommen (voor toekomstige uitbreiding)

### sale_date (Date)
- **Type**: Date (YYYY-MM-DD)
- **Voorbeeld**: `2024-01-15`
- **Beschrijving**: Verkoopdatum (als verkocht)

### sale_price (Numeric)
- **Type**: Number (Euro)
- **Voorbeeld**: `9800000`
- **Beschrijving**: Uiteindelijke verkoopprijs (als verkocht)

### year_built (Numeric)
- **Type**: Number (jaar)
- **Voorbeeld**: `2015`
- **Beschrijving**: Bouwjaar van het property

### occupancy_rate (Numeric)
- **Type**: Number (0-1)
- **Voorbeeld**: `0.95`
- **Beschrijving**: Bezetting percentage

## Data Kwaliteit Checklist

Voor elke rij controleer:
- [ ] Alle verplichte kolommen ingevuld
- [ ] Geen lege cellen in belangrijke velden
- [ ] Numerieke waarden zijn getallen (geen tekst)
- [ ] Decimale getallen gebruiken punten (0.065 niet 0,065)
- [ ] sold_within_180d is 0 of 1 (niet "ja"/"nee")
- [ ] Realistische waarden (geen 0 of extreme outliers)
- [ ] Categorische waarden zijn consistent (Rotterdam, niet rotterdam)

## Voorbeeld Data

Zie `DATA_COLLECTION_TEMPLATE.csv` voor voorbeeld rijen.

## Vragen?

Neem contact op als je vragen hebt over:
- Welke data je nodig hebt
- Hoe bepaalde waarden te bepalen
- Data kwaliteit issues







