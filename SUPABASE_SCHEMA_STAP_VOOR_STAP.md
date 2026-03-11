# Supabase Schema Aanmaken - Stap voor Stap

## Stap 3: Database Schema Aanmaken (Duidelijke Uitleg)

### Optie 1: Via Supabase SQL Editor (Aanbevolen)

1. **Open Supabase Dashboard:**
   - Ga naar https://supabase.com/dashboard
   - Selecteer je project

2. **Open SQL Editor:**
   - Klik in de linker sidebar op **"SQL Editor"** (icoon: </>)
   - Je ziet nu een leeg scherm met een tekstveld bovenaan

3. **Open het schema.sql bestand:**
   - Open op je computer: `ml-service/supabase/schema.sql`
   - Selecteer ALLES (Cmd+A / Ctrl+A)
   - Kopieer alles (Cmd+C / Ctrl+C)

4. **Plak in SQL Editor:**
   - Ga terug naar Supabase SQL Editor
   - Klik in het grote tekstveld (waar je queries kunt typen)
   - Plak de SQL code (Cmd+V / Ctrl+V)
   - Je ziet nu alle SQL statements

5. **Run de Query:**
   - Klik op de groene **"Run"** knop (rechtsboven, of gebruik Cmd+Enter / Ctrl+Enter)
   - Wacht even... je ziet onderaan "Success. No rows returned" of vergelijkbaar

6. **Verifieer dat het werkt:**
   - Klik in de linker sidebar op **"Table Editor"**
   - Je zou nu moeten zien:
     - ✅ `customers`
     - ✅ `properties`
     - ✅ `demo_cases`
     - ✅ `training_runs`
     - ✅ `api_usage`

### Optie 2: Via Table Editor (Handmatig - Niet Aanbevolen)

Als je liever handmatig tabellen aanmaakt (veel meer werk):

1. Ga naar **Table Editor**
2. Klik **"New Table"**
3. Maak elke tabel één voor één aan...

**Maar dit is veel meer werk! Gebruik Optie 1.**

## Wat gebeurt er precies?

Het SQL script maakt 5 tabellen aan:
1. **customers** - Voor klanten/clienten
2. **properties** - Voor vastgoed listings
3. **training_runs** - Voor training geschiedenis
4. **api_usage** - Voor API tracking
5. **demo_cases** - Voor demo cases

Plus:
- Indexes voor snellere queries
- Triggers voor automatische timestamp updates
- Foreign keys voor data integriteit

## Troubleshooting

**"Error: relation already exists":**
- Tabellen bestaan al - dat is OK! Het script gebruikt `CREATE TABLE IF NOT EXISTS`
- Je kunt gewoon doorgaan

**"Permission denied":**
- Check of je de juiste rechten hebt in je Supabase project
- Je moet project owner zijn

**"Syntax error":**
- Check of je het hele bestand hebt gekopieerd
- Check of er geen extra tekst is toegevoegd

## Screenshot Beschrijving

**SQL Editor ziet er zo uit:**
```
┌─────────────────────────────────────────┐
│  SQL Editor                              │
├─────────────────────────────────────────┤
│  [Groot tekstveld voor SQL code]        │
│                                          │
│  CREATE TABLE IF NOT EXISTS customers (│
│    id SERIAL PRIMARY KEY,                │
│    ...                                   │
│  );                                      │
│                                          │
│  [Run] [Save] [New Query]               │
└─────────────────────────────────────────┘
```

**Na Run zie je:**
```
┌─────────────────────────────────────────┐
│  Success. No rows returned              │
│  Query executed in 0.05s                 │
└─────────────────────────────────────────┘
```

## Volgende Stap

Na het aanmaken van het schema:
```bash
python scripts/test_supabase_connection.py
```

Dit test of alles werkt!




