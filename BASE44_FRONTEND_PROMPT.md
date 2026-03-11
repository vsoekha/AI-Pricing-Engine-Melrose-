# Base44 Frontend Prompt: Vastgoed AI Analyse Demo

## Overview
Build a professional, demo-ready frontend for a Dutch real estate AI pricing analysis tool. The application has two main categories:
1. **Bestaand Vastgoed** (Existing Assets) - AI-powered price optimization
2. **Projectontwikkeling** (Development Projects) - Financial scenario analysis

## API Base URL
```
https://your-render-app.onrender.com
```
Or for local development:
```
http://localhost:8000
```

---

## UI Structure

### Main Layout
- **Header**: "Vastgoed AI Analyse" with logo/icon
- **Tab Navigation**: Two tabs - "Bestaand Vastgoed" and "Projectontwikkeling"
- **Form Section**: Input fields for the selected category
- **Results Section**: Display comprehensive analysis results
- **Footer**: Simple footer with version info

### Design Requirements
- Clean, professional, modern design
- Use Dutch language throughout
- Responsive (works on desktop and tablet)
- Color scheme: Professional blues/grays with accent colors for important metrics
- Loading states when API calls are in progress
- Error handling with user-friendly messages

---

## Category 1: Bestaand Vastgoed (Existing Assets)

### Form Fields (Dutch Labels)

Create a form with the following fields:

1. **Asset Type** (Dropdown)
   - Label: "Asset Type"
   - Options: `logistics`, `office`, `resi`, `retail`, `mixed`
   - Placeholder: "Selecteer asset type"

2. **City** (Text Input)
   - Label: "Stad"
   - Placeholder: "Bijv. Rotterdam, Amsterdam, Utrecht"

3. **Size (m²)** (Number Input)
   - Label: "Oppervlakte (m²)"
   - Min: 0
   - Placeholder: "Bijv. 12000"

4. **Quality Score** (Number Input, Slider, or Number)
   - Label: "Kwaliteitsscore"
   - Min: 0, Max: 1, Step: 0.01
   - Placeholder: "0.00 - 1.00"
   - Helper text: "Score van 0 (laag) tot 1 (hoog)"

5. **NOI Annual** (Number Input)
   - Label: "Netto Operationeel Inkomen (jaarlijks)"
   - Min: 0
   - Placeholder: "Bijv. 620000"
   - Format: Display with thousand separators (e.g., 620.000)

6. **Cap Rate Market** (Number Input)
   - Label: "Cap Rate Markt"
   - Min: 0, Max: 1, Step: 0.001
   - Placeholder: "Bijv. 0.065"
   - Helper text: "Als percentage: 0.065 = 6.5%"

7. **Interest Rate** (Number Input)
   - Label: "Rente"
   - Min: 0, Max: 1, Step: 0.001
   - Placeholder: "Bijv. 0.025"
   - Helper text: "Als percentage: 0.025 = 2.5%"

8. **Liquidity Index** (Number Input, Slider)
   - Label: "Liquiditeitsindex"
   - Min: 0, Max: 1, Step: 0.01
   - Placeholder: "0.00 - 1.00"
   - Helper text: "0 = laag, 1 = hoog"

9. **List Price** (Number Input)
   - Label: "Vraagprijs"
   - Min: 0
   - Placeholder: "Bijv. 9500000"
   - Format: Display with thousand separators

10. **Comparable Median Price** (Number Input)
    - Label: "Mediaan Prijs Vergelijkbare Objecten"
    - Min: 0
    - Placeholder: "Bijv. 9900000"
    - Format: Display with thousand separators

### Submit Button
- Label: "Analyseer Prijs"
- Loading state: Show spinner and disable button during API call

### API Call

**Endpoint**: `POST /recommend_price`

**Request Body**:
```json
{
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
}
```

**Response Structure**:
```json
{
  "base_price": 9500000,
  "base_sale_probability": 0.7234,
  "recommended_price": 9785000,
  "recommended_sale_probability": 0.7891,
  "expected_uplift": 285000,
  "price_range_up": 8.5,
  "price_range_down": 5.2,
  "price_sensitivity": "gemiddeld",
  "optimal_timing": "Q1 2026",
  "timing_reason": "Gunstige marktomstandigheden...",
  "timing_impact": 8.5,
  "buyer_demand_score": 7.8,
  "buyer_demand_level": "hoog",
  "buyer_profiles": ["Pensioenfondsen", "Institutionele investeerders"],
  "scenarios": [
    {
      "scenario_name": "Basis Scenario",
      "interest_rate": 0.025,
      "recommended_price": 9785000,
      "sale_probability": 0.7891,
      "expected_revenue": 7720000,
      "value_impact": 0.0
    },
    {
      "scenario_name": "Optimistisch (rente -0.5%)",
      "interest_rate": 0.02,
      "recommended_price": 9850000,
      "sale_probability": 0.8123,
      "expected_revenue": 7990000,
      "value_impact": 3.5
    }
  ],
  "market_liquidity": "Hoog - Actieve markt met veel kopers",
  "market_trend": "Bullish - Sterke vraag, lage cap rates",
  "comparable_yield": 0.065
}
```

### Results Display (Category 1)

Display results in a clean, organized layout with the following sections:

#### 1. Core Pricing (Highlighted Card)
- **Huidige Vraagprijs**: `base_price` (formatted with € and thousand separators)
- **Verkoopkans Huidige Prijs**: `base_sale_probability` as percentage (e.g., "72.3%")
- **Aanbevolen Prijs**: `recommended_price` (formatted, highlighted in green if higher)
- **Verkoopkans Aanbevolen Prijs**: `recommended_sale_probability` as percentage
- **Verwachte Opbrengstverhoging**: `expected_uplift` (formatted, show in green if positive)

#### 2. Price Sensitivity (Card)
- **Prijsgevoeligheid**: `price_sensitivity` (with color: laag=green, gemiddeld=yellow, hoog=red)
- **Bereik Omhoog**: `price_range_up` + "%"
- **Bereik Omlaag**: `price_range_down` + "%"
- Visual: Show a range bar or gauge

#### 3. Timing Recommendation (Card)
- **Optimale Timing**: `optimal_timing`
- **Reden**: `timing_reason`
- **Impact**: `timing_impact` + "%"

#### 4. Buyer Demand (Card)
- **Koper Vraag Score**: `buyer_demand_score` / 10 (show as progress bar or gauge)
- **Niveau**: `buyer_demand_level` (with color coding)
- **Interessante Kopers**: List `buyer_profiles` as tags/badges

#### 5. Scenario Analysis (Table/Cards)
Display all scenarios in a table or card grid:
- Scenario Name
- Interest Rate (as %)
- Recommended Price (€)
- Sale Probability (%)
- Expected Revenue (€)
- Value Impact (%)
- Color code: positive = green, negative = red

#### 6. Market Insights (Card)
- **Marktliquiditeit**: `market_liquidity`
- **Markttrend**: `market_trend` (with icon: bullish=↑, bearish=↓, neutral=→)
- **Vergelijkbaar Rendement**: `comparable_yield` as percentage

#### 7. Waarop is dit advies gebaseerd? (Data-bronnen – essentieel voor demo-waarde)
Toon een compact blok dat de koppeling met realtime/live datastromen expliciet maakt. Doel: tijdens de demo laten zien dat de prijs in de live situatie gevoed wordt door marktdata.

- **Cap rate** → *In live: marktdata (CBRE/JLL cap rate survey, BAR). Nu: uw invoer.*
- **Rente** → *In live: ECB / rente-feed. Nu: uw invoer.*
- **Comps / vergelijkbare prijs** → *In live: transactiedata (NVM Business, RealNext). Nu: uw invoer.*
- **Marktliquiditeit** → *In live: transactievolume, opname. Nu: uw inschatting (liquiditeitsindex).*
- **Verkoopkans (AI)** → *Getraind op historische transacties; in live aangevuld met actuele comps en marktdata.*

Visueel: kleine info-card of uitklapbare sectie, titel "Waarop is dit advies gebaseerd?" of "Data-bronnen (live)". Optioneel: icoon "live" of "in productie" bij velden die in live uit externe bronnen komen.

---

## Category 2: Projectontwikkeling (Development Projects)

### Form Fields (Dutch Labels)

1. **Location** (Text Input)
   - Label: "Locatie"
   - Placeholder: "Bijv. Amsterdam, Rotterdam"

2. **Project Type** (Dropdown)
   - Label: "Project Type"
   - Options: `residential`, `mixed-use`, `commercial`, `logistics`
   - Placeholder: "Selecteer project type"

3. **Units Count** (Number Input)
   - Label: "Aantal Units"
   - Min: 1
   - Placeholder: "Bijv. 50"

4. **Total Area (m²)** (Number Input)
   - Label: "Totale Oppervlakte (m²)"
   - Min: 0
   - Placeholder: "Bijv. 5000"

5. **Expected Sale Price per m²** (Number Input)
   - Label: "Verwachte Verkoopprijs per m²"
   - Min: 0
   - Placeholder: "Bijv. 6000"
   - Format: Display with thousand separators

6. **Build Cost per m²** (Number Input)
   - Label: "Bouwkosten per m²"
   - Min: 0
   - Placeholder: "Bijv. 3500"
   - Format: Display with thousand separators

7. **Soft Cost %** (Number Input)
   - Label: "Soft Costs Percentage"
   - Min: 0, Max: 1, Step: 0.01
   - Placeholder: "Bijv. 0.12"
   - Helper text: "0.12 = 12%"

8. **Contingency %** (Number Input)
   - Label: "Contingency Percentage"
   - Min: 0, Max: 1, Step: 0.01
   - Placeholder: "Bijv. 0.07"
   - Helper text: "0.07 = 7%"

9. **Land Cost** (Number Input)
   - Label: "Grondkosten"
   - Min: 0
   - Placeholder: "Bijv. 5000000"
   - Format: Display with thousand separators

10. **Duration (Months)** (Number Input)
    - Label: "Projectduur (maanden)"
    - Min: 1
    - Placeholder: "Bijv. 24"

11. **Interest Rate** (Number Input)
    - Label: "Financieringsrente"
    - Min: 0, Max: 1, Step: 0.001
    - Placeholder: "Bijv. 0.045"
    - Helper text: "0.045 = 4.5%"

12. **Target Margin %** (Number Input)
    - Label: "Doelmarge Percentage"
    - Min: 0, Max: 1, Step: 0.01
    - Placeholder: "Bijv. 0.18"
    - Helper text: "0.18 = 18%"

13. **Exit Cap Rate** (Number Input, Optional)
    - Label: "Exit Cap Rate (optioneel)"
    - Min: 0, Max: 1, Step: 0.001
    - Placeholder: "Bijv. 0.055"

14. **Absorption Months** (Number Input, Optional)
    - Label: "Absorptie Periode (maanden, optioneel)"
    - Min: 1
    - Placeholder: "Bijv. 12"

### Submit Button
- Label: "Analyseer Project"
- Loading state: Show spinner and disable button during API call

### API Call

**Endpoint**: `POST /dev_project_analysis`

**Request Body**:
```json
{
  "location": "Amsterdam",
  "project_type": "residential",
  "units_count": 50,
  "total_area_m2": 5000,
  "expected_sale_price_per_m2": 6000,
  "build_cost_per_m2": 3500,
  "soft_cost_pct": 0.12,
  "contingency_pct": 0.07,
  "land_cost": 5000000,
  "duration_months": 24,
  "interest_rate": 0.045,
  "target_margin_pct": 0.18
}
```

**Response Structure**:
```json
{
  "project_summary": {
    "location": "Amsterdam",
    "project_type": "residential",
    "units_count": 50,
    "total_area_m2": 5000,
    "duration_months": 24
  },
  "base_scenario": {
    "scenario_name": "Basis Scenario",
    "total_revenue": 30000000,
    "total_project_cost": 24500000,
    "profit": 5500000,
    "margin": 0.1833,
    "meets_target_margin": true
  },
  "downside_scenario": {
    "scenario_name": "Negatief Scenario (-7% verkoopprijs, +7% bouwkosten, +0.5% rente)",
    "total_revenue": 27900000,
    "total_project_cost": 26200000,
    "profit": 1700000,
    "margin": 0.0609,
    "meets_target_margin": false
  },
  "upside_scenario": {
    "scenario_name": "Positief Scenario (+5% verkoopprijs, -3% bouwkosten, -0.5% rente)",
    "total_revenue": 31500000,
    "total_project_cost": 23800000,
    "profit": 7700000,
    "margin": 0.2444,
    "meets_target_margin": true
  },
  "recommended_action": "DOORGAAN",
  "recommendation_reason": "Basis scenario voldoet aan doelmarge en negatief scenario blijft acceptabel..."
}
```

### Results Display (Category 2)

Display results in a clean, organized layout:

#### 1. Project Summary (Card)
- Location
- Project Type
- Units Count
- Total Area (m²)
- Duration (months)

#### 2. Recommended Action (Highlighted Card)
- **Aanbeveling**: `recommended_action` (large, color-coded)
  - DOORGAAN = Green
  - HERZIEN = Yellow
  - HERPRIJZEN = Orange
  - WACHTEN = Red
- **Reden**: `recommendation_reason`

#### 3. Scenario Comparison (Three Cards or Table)

For each scenario (Base, Downside, Upside), display:

**Scenario Name** (as header)

- **Totale Opbrengst**: `total_revenue` (€, formatted)
- **Totale Projectkosten**: `total_project_cost` (€, formatted)
- **Winst**: `profit` (€, formatted, color: green if positive, red if negative)
- **Marge**: `margin` as percentage (e.g., "18.33%")
- **Doelmarge Gehaald**: ✓ or ✗ (with color)

Visual: Show profit/margin as a progress bar or gauge. Use color coding (green for good, red for bad).

#### 4. Financial Breakdown (Optional - Detailed View)
For the base scenario, show:
- Hard Costs = `total_area_m2 × build_cost_per_m2`
- Soft Costs = Hard Costs × `soft_cost_pct`
- Contingency = Hard Costs × `contingency_pct`
- Land Cost
- Financing Cost
- Total Project Cost
- Total Revenue
- Profit
- Margin

#### 5. Waarop is dit advies gebaseerd? (Data-bronnen – essentieel voor demo-waarde)
Toon een compact blok dat de koppeling met realtime/live datastromen voor development expliciet maakt.

- **Verkoopprijs per m²** → *In live: vergelijkbare nieuwbouwtransacties (comps). Nu: uw invoer.*
- **Bouwkosten / stichtingskosten** → *In live: begrotingen, bouwkostenindex. Nu: uw invoer.*
- **Financieringsrente** → *In live: ECB / rente-feed. Nu: uw invoer.*
- **Absorptie / vraag** → *In live: woningbehoefte, verkooptempo. Nu: optionele invoer.*

Visueel: kleine info-card, titel "Waarop is dit advies gebaseerd?" of "Data-bronnen (live)".

---

## Error Handling

### API Errors
- Display user-friendly error messages in Dutch
- Show error state in a red alert/notification box
- Example: "Er is een fout opgetreden bij het analyseren. Probeer het opnieuw."

### Validation Errors
- Show inline validation errors for each field
- Highlight invalid fields in red
- Display helpful error messages (e.g., "Dit veld is verplicht", "Waarde moet tussen 0 en 1 zijn")

### Loading States
- Show loading spinner during API calls
- Disable submit button during loading
- Show "Analyseren..." text

---

## Demo Features

### Pre-fill Example Data
Add a dropdown or buttons **"Kies voorbeeld"** met 4 voorbeelden per categorie. Bij selectie vult het formulier automatisch in. Data komt van `GET /demo_cases?category=category_1` of `category_2`; gebruik `case_data` van de geselecteerde case.

#### Voorbeelden Category 1 (Bestaand Vastgoed) – 4 cases:

1. **Distributiecentrum Rotterdam Waalhaven** – logistiek, sterke markt
2. **Kantoor Zuidas Amsterdam** – kantoor, uitdagende markt
3. **Winkelstraat Retail Eindhoven** – retail, stabiele cashflow
4. **Appartementencomplex Utrecht** – multi-family, stabiel rendement

#### Voorbeelden Category 2 (Projectontwikkeling) – 4 cases:

1. **Woningbouw Amsterdam Sloterdijk** – residentieel, winstgevend
2. **Mixed-Use Rotterdam Zuid** – gemengd gebruik
3. **Logistiek Hall Tilburg** – distributie, build-to-sell
4. **Kantoorontwikkeling Den Haag** – kantoor, risicovol

#### Fallback (als API niet bereikbaar) – Example Data (Category 1):
```javascript
// Case 1 – Distributiecentrum Rotterdam
{ asset_type: "logistics", city: "Rotterdam", size_m2: 12000, quality_score: 0.85, noi_annual: 624000, cap_rate_market: 0.062, interest_rate: 0.035, liquidity_index: 0.78, list_price: 9900000, comp_median_price: 10200000 }

// Case 2 – Kantoor Amsterdam
{ asset_type: "office", city: "Amsterdam", size_m2: 5200, quality_score: 0.72, noi_annual: 325000, cap_rate_market: 0.058, interest_rate: 0.035, liquidity_index: 0.52, list_price: 5850000, comp_median_price: 5600000 }

// Case 3 – Retail Eindhoven
{ asset_type: "retail", city: "Eindhoven", size_m2: 1800, quality_score: 0.68, noi_annual: 162000, cap_rate_market: 0.065, interest_rate: 0.035, liquidity_index: 0.62, list_price: 2480000, comp_median_price: 2500000 }

// Case 4 – Multi-family Utrecht
{ asset_type: "resi", city: "Utrecht", size_m2: 2800, quality_score: 0.76, noi_annual: 298000, cap_rate_market: 0.042, interest_rate: 0.035, liquidity_index: 0.70, list_price: 6950000, comp_median_price: 7100000 }
```

#### Fallback (als API niet bereikbaar) – Example Data (Category 2):
```javascript
// Case 1 – Woningbouw Amsterdam
{ location: "Amsterdam", project_type: "residential", units_count: 54, total_area_m2: 4860, expected_sale_price_per_m2: 6200, build_cost_per_m2: 3200, soft_cost_pct: 0.12, contingency_pct: 0.07, land_cost: 4200000, duration_months: 28, interest_rate: 0.045, target_margin_pct: 0.18 }

// Case 2 – Mixed-Use Rotterdam
{ location: "Rotterdam", project_type: "mixed-use", units_count: 24, total_area_m2: 7200, expected_sale_price_per_m2: 5200, build_cost_per_m2: 3600, soft_cost_pct: 0.14, contingency_pct: 0.08, land_cost: 6500000, duration_months: 32, interest_rate: 0.050, target_margin_pct: 0.16 }

// Case 3 – Logistiek Tilburg
{ location: "Tilburg", project_type: "logistics", units_count: 1, total_area_m2: 8000, expected_sale_price_per_m2: 1450, build_cost_per_m2: 950, soft_cost_pct: 0.10, contingency_pct: 0.06, land_cost: 1200000, duration_months: 14, interest_rate: 0.045, target_margin_pct: 0.15 }

// Case 4 – Kantoor Den Haag
{ location: "Den Haag", project_type: "commercial", units_count: 1, total_area_m2: 6000, expected_sale_price_per_m2: 4200, build_cost_per_m2: 3850, soft_cost_pct: 0.16, contingency_pct: 0.09, land_cost: 5800000, duration_months: 36, interest_rate: 0.052, target_margin_pct: 0.12 }
```

### Live ECB-rente (eerste live datastroom)

Voeg een **live rente-blok** toe op de pagina voor *Bestaand Vastgoed* op basis van de nieuwe backend endpoints:

- **Endpoint (GET)**: `/ecb/main_refi_rate`  
  - Haal bij initial load van de app (of bij wisselen naar tab "Bestaand Vastgoed") de actuele ECB main refinancing operations rate op.  
  - Toon een kleine kaart of badge boven of naast het veld **Rente**, met tekst zoals:  
    - "ECB beleidsrente (live): 3,50% – datum: 2026-02-20"  
    - "Bron: ECB Data Portal"  
  - Sla de waarde op in state, bijv. `ecbRate`.

- **Form-veld koppeling**:  
  - Vul het veld **Rente** standaard voor met de opgehaalde `ecbRate` (bijv. 0.035).  
  - Laat de gebruiker dit veld aanpassen indien gewenst (bijvoorbeeld label: "Rente (vooraf ingevuld via ECB, aanpasbaar)").

- **Toggle voor backend-logica (optioneel maar wenselijk)**:  
  - Voeg een schakelaar toe: "Gebruik live ECB‑rente in berekening" (default: **aan**).  
  - Als deze aan staat, wordt de query-parameter `gebruik_ecb_rente=true` toegevoegd aan de call naar `/recommend_price`.  
  - Als deze uit staat, wordt `gebruik_ecb_rente=false` of wordt de parameter weggelaten, en gebruikt de backend de ingevulde rente in het formulier.

### Voorbeeld-flow in de UI (samengevat)

1. App laadt → doe `GET /ecb/main_refi_rate` → zet `ecbRate` in state.  
2. Vul **Rente** voor met `ecbRate` en toon een info-card met de live rente + datum + bron.  
3. Bij klikken op "Analyseer Prijs": stuur de POST naar `/recommend_price` met de juiste query-parameter:
   - Met live rente: `POST /recommend_price?gebruik_ecb_rente=true`  
   - Zonder live rente: `POST /recommend_price` (of `?gebruik_ecb_rente=false`)

---

## Technical Requirements

### API Integration
- Use `fetch()` or `axios` for API calls
- Handle CORS (API already configured to allow all origins)
- Set `Content-Type: application/json` header
- Handle network errors gracefully

### Number Formatting
- Format all currency values with € symbol and thousand separators (e.g., €9.500.000)
- Format percentages with % symbol (e.g., 72.3%)
- Format decimals appropriately (2 decimals for currency, 1-2 for percentages)

### State Management
- Manage form state (controlled inputs)
- Manage loading state
- Manage results state
- Manage error state

### Responsive Design
- Mobile-friendly (at least tablet size)
- Cards stack vertically on smaller screens
- Tables scroll horizontally on mobile if needed

---

## Example Code Structure (Pseudo-code)

```javascript
// State
const [activeTab, setActiveTab] = useState('existing');
const [formData, setFormData] = useState({});
const [loading, setLoading] = useState(false);
const [results, setResults] = useState(null);
const [error, setError] = useState(null);

// API Call
const handleSubmit = async () => {
  setLoading(true);
  setError(null);
  
  try {
    const endpoint = activeTab === 'existing' 
      ? '/recommend_price' 
      : '/dev_project_analysis';
    
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(formData)
    });
    
    if (!response.ok) throw new Error('API Error');
    
    const data = await response.json();
    setResults(data);
  } catch (err) {
    setError(err.message);
  } finally {
    setLoading(false);
  }
};

// Format currency
const formatCurrency = (value) => {
  return new Intl.NumberFormat('nl-NL', {
    style: 'currency',
    currency: 'EUR'
  }).format(value);
};

// Format percentage
const formatPercent = (value) => {
  return `${(value * 100).toFixed(1)}%`;
};
```

---

## Design Inspiration

- Clean, professional dashboard style
- Use cards/sections to organize information
- Color coding for metrics (green = good, red = bad, yellow = warning)
- Icons for visual clarity (€, %, ↑, ↓, ✓, ✗)
- Smooth transitions and animations
- Professional typography

---

## Deliverables

1. Complete working frontend with both categories
2. All form fields with proper validation
3. API integration for both endpoints
4. Comprehensive results display
5. Error handling and loading states
6. Responsive design
7. Dutch language throughout
8. Demo-ready with example data button

---

## Notes

- All API responses are in Dutch
- All field labels should be in Dutch
- All error messages should be in Dutch
- The API base URL should be configurable (environment variable or config)
- The API already has CORS enabled for all origins



