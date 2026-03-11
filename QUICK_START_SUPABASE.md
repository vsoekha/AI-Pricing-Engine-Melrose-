# Quick Start: Supabase Setup

## Stap 1: Credentials Ophalen (2 minuten)

1. **Ga naar Supabase Dashboard:**
   - https://supabase.com/dashboard
   - Selecteer je project

2. **Settings → API:**
   - Kopieer **Project URL** → `SUPABASE_URL`
   - Kopieer **service_role key** (niet anon key!) → `SUPABASE_KEY`

3. **Maak `.env` bestand:**
   ```bash
   cp .env.example .env
   ```
   
4. **Vul `.env` in:**
   ```bash
   SUPABASE_URL=https://xxxxx.supabase.co
   SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   ```

## Stap 2: Database Schema Aanmaken (1 minuut)

1. **Open Supabase Dashboard → SQL Editor**

2. **Open `supabase/schema.sql` en kopieer alle SQL**

3. **Plak in SQL Editor en klik "Run"**

4. **Check of tabellen zijn aangemaakt:**
   - Table Editor → zie je: `customers`, `properties`, `demo_cases`, etc.?

## Stap 3: Install Dependencies (1 minuut)

```bash
cd ~/Desktop/ml-service
source .venv/bin/activate
pip install supabase python-dotenv
```

## Stap 4: Test Connection (30 seconden)

```bash
python scripts/test_supabase_connection.py
```

Als alles werkt zie je: `✓ ALL TESTS PASSED!`

## Stap 5: Migreer Data (Optioneel)

Als je al data in SQLite hebt:

```bash
python scripts/migrate_to_supabase.py
```

## Stap 6: Start API

```bash
uvicorn api.main:app --reload
```

De API gebruikt nu automatisch Supabase als de credentials zijn ingesteld!

## Render Deployment

Voeg deze environment variables toe in Render:

1. Ga naar je service in Render
2. Settings → Environment
3. Voeg toe:
   - `SUPABASE_URL` = je project URL
   - `SUPABASE_KEY` = je service_role key

## Troubleshooting

**"Supabase credentials not found":**
- Check of `.env` bestaat en correct is ingevuld
- Check of je `python-dotenv` hebt geïnstalleerd

**"Table not found":**
- Run het schema.sql script in Supabase SQL Editor
- Check of tabellen bestaan in Table Editor

**"Connection refused":**
- Check of je IP toegang heeft (Supabase → Settings → Database)
- Gebruik connection pooling als je buiten Supabase netwerk bent

## Klaar! 🎉

Je database draait nu in Supabase. Alle demo cases en data worden opgeslagen in de cloud.




