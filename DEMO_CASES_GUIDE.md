# Demo Cases Guide - Nederlandse Terminologie

## Demo Cases Systeem

### Wat zijn Demo Cases?

Demo cases zijn vooraf opgeslagen voorbeelden die je kunt gebruiken tijdens presentaties. Ze bevatten realistische data voor verschillende scenario's.

### Demo Cases Endpoints

**Lijst alle cases:**
```
GET /demo_cases
GET /demo_cases?category=category_1  (filter op category)
```

**Haal specifieke case op:**
```
GET /demo_cases/{case_id}
```

**Sla nieuwe case op:**
```
POST /demo_cases
{
  "case_name": "Naam van de case",
  "case_type": "existing_asset" of "development_project",
  "category": "category_1" of "category_2",
  "case_data": { ... volledige request body ... },
  "description": "Beschrijving"
}
```

**Verwijder case:**
```
DELETE /demo_cases/{case_id}
```

## Vooraf Aangemaakte Demo Cases

Alle cases zijn gebaseerd op realistische NL-marktdata (cap rates, bouwkosten, verkoopprijzen per m²).

### Category 1: Bestaand Vastgoed (4 cases)

1. **Distributiecentrum Rotterdam Waalhaven**
   - Logistiek 12.000 m², sterke markt, hoge vraag (logistiek ~30% beleggingsvolume NL)
   - Laat zien: optimale prijs, timing, buyer demand

2. **Kantoor Zuidas Amsterdam**
   - Kantoor 5.200 m², uitdagende kantorenmarkt (lage opname 2024)
   - Laat zien: scenario analysis, price sensitivity, timing

3. **Winkelstraat Retail Eindhoven**
   - Retail 1.800 m², drukke straat, stabiele cashflow
   - Laat zien: gebalanceerde aanbevelingen, marktliquiditeit

4. **Appartementencomplex Utrecht**
   - Multi-family 35 eenheden, 2.800 m², volverhuurd, BAR 4,2%
   - Laat zien: stabiel rendement, resi-yields

### Category 2: Projectontwikkeling (4 cases)

1. **Woningbouw Amsterdam Sloterdijk**
   - 54 appartementen, build-to-sell, sterke locatie
   - Laat zien: DOORGAAN, winstgevend scenario

2. **Mixed-Use Rotterdam Zuid**
   - Kantoren + retail + wonen, 7.200 m²
   - Laat zien: scenario vergelijking, hogere complexiteit

3. **Logistiek Hall Tilburg**
   - Distributiehal 8.000 m², build-to-sell, korte doorlooptijd
   - Laat zien: logistiek development, andere koststructuur

4. **Kantoorontwikkeling Den Haag**
   - Nieuw kantoor 6.000 m², risicovolle kantorenmarkt
   - Laat zien: HERZIEN/WACHTEN, hoge kosten, lage marge

## Nederlandse Terminologie

### Category 1 - Bestaand Vastgoed

**Asset Types:**
- `logistics` → "Logistiek"
- `office` → "Kantoor"
- `resi` → "Residentieel"
- `retail` → "Retail"
- `mixed` → "Gemengd"

**Response Velden:**
- `base_price` → "Huidige Vraagprijs"
- `recommended_price` → "Aanbevolen Prijs"
- `base_sale_probability` → "Verkoopkans Huidige Prijs"
- `expected_uplift` → "Verwachte Extra Opbrengst"
- `price_range_up` → "Maximale Ruimte Omhoog (%)"
- `price_range_down` → "Maximale Ruimte Omlaag (%)"
- `price_sensitivity` → "Prijsgevoeligheid" (laag/gemiddeld/hoog)
- `optimal_timing` → "Optimale Timing"
- `buyer_demand_score` → "Koper Vraag Score (0-10)"
- `buyer_demand_level` → "Koper Vraag Niveau" (laag/gemiddeld/hoog)
- `market_liquidity` → "Markt Liquiditeit"
- `market_trend` → "Markt Trend"

### Category 2 - Projectontwikkeling

**Project Types:**
- `residential` → "Residentieel"
- `mixed-use` → "Gemengd Gebruik"
- `commercial` → "Commercieel"
- `office` → "Kantoor"
- `logistics` → "Logistiek"

**Response Velden:**
- `total_revenue` → "Totale Opbrengst"
- `total_project_cost` → "Totale Projectkosten"
- `profit` → "Winst"
- `margin` → "Marge (%)"
- `meets_target_margin` → "Voldoet aan Doelmarge"
- `recommended_action` → "Aanbevolen Actie" (DOORGAAN/HERZIEN/HERPRIJZEN/WACHTEN)

**Scenarios:**
- `base_scenario` → "Basis Scenario"
- `downside_scenario` → "Negatief Scenario"
- `upside_scenario` → "Positief Scenario"

## Demo Cases Gebruiken

### In Frontend (Base44)

1. **Laad demo cases:**
   ```javascript
   const cases = await fetch('/demo_cases?category=category_1')
     .then(r => r.json());
   ```

2. **Toon case selector:**
   - Dropdown of lijst met beschikbare cases
   - Gebruiker selecteert case
   - Vul formulier automatisch in

3. **Submit case:**
   - Gebruik case_data als request body
   - Stuur naar juiste endpoint

### Voor Presentaties

1. **Voorbereiden:**
   - Maak demo cases aan met `scripts/create_demo_cases.py`
   - Test alle cases lokaal

2. **Tijdens demo:**
   - Selecteer case uit dropdown
   - Formulier vult automatisch in
   - Klik "Analyseer"
   - Laat resultaten zien

3. **Switch tussen cases:**
   - Verschillende cases tonen verschillende insights
   - Laat zien hoe AI reageert op verschillende inputs

## Demo Cases Aanmaken

```bash
cd ~/Desktop/ml-service
source .venv/bin/activate
python scripts/create_demo_cases.py
```

Dit maakt 8 vooraf gedefinieerde cases aan (4 per category).

## Custom Demo Cases

Je kunt ook via de API cases aanmaken:

```bash
curl -X POST "http://localhost:8000/demo_cases" \
  -H "Content-Type: application/json" \
  -d '{
    "case_name": "Mijn Custom Case",
    "case_type": "existing_asset",
    "category": "category_1",
    "case_data": {
      "asset_type": "logistics",
      "city": "Eindhoven",
      ...
    },
    "description": "Beschrijving van deze case"
  }'
```

## Best Practices

1. **Case Namen:** Gebruik duidelijke, beschrijvende namen
2. **Realistische Data:** Gebruik realistische waarden
3. **Diverse Cases:** Verschillende scenario's (goed/slecht/gemiddeld)
4. **Beschrijvingen:** Leg uit wat de case laat zien
5. **Test Cases:** Test alle cases voordat je presenteert




