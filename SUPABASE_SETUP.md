# Supabase Setup Guide

## Stap 1: Supabase Credentials Ophalen

1. **Ga naar je Supabase project dashboard:**
   - https://supabase.com/dashboard
   - Selecteer je project

2. **Ga naar Settings → API:**
   - Kopieer je **Project URL** (bijv. `https://xxxxx.supabase.co`)
   - Kopieer je **anon/public key** (voor client-side)
   - Kopieer je **service_role key** (voor server-side - BEVEILIG DIT!)

3. **Ga naar Settings → Database:**
   - Kopieer je **Connection String** (URI format)
   - Of gebruik: `postgresql://postgres:[PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres`

## Stap 2: Environment Variables Instellen

Maak een `.env` bestand in de root van je project:

```bash
# Supabase Configuration
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...  # service_role key
SUPABASE_DB_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres

# Of gebruik direct PostgreSQL connection
DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres
```

**BELANGRIJK:**
- Voeg `.env` toe aan `.gitignore` (niet committen!)
- Gebruik `service_role` key voor backend (niet `anon` key)
- Voor Render deployment: voeg environment variables toe in Render dashboard

## Stap 3: Database Schema Aanmaken

Gebruik het SQL script in `supabase/schema.sql` om tabellen aan te maken.

**Opties:**
1. **Via Supabase Dashboard:**
   - Ga naar SQL Editor
   - Plak het SQL script
   - Run

2. **Via Python script:**
   ```bash
   python scripts/setup_supabase.py
   ```

## Stap 4: Data Migreren (Optioneel)

Als je al data in SQLite hebt:

```bash
python scripts/migrate_to_supabase.py
```

## Stap 5: Testen

```bash
python scripts/test_supabase_connection.py
```

## Render Deployment

Voeg deze environment variables toe in Render:

1. Ga naar je service in Render
2. Settings → Environment
3. Voeg toe:
   - `SUPABASE_URL`
   - `SUPABASE_KEY` (service_role)
   - `DATABASE_URL` (PostgreSQL connection string)

## Troubleshooting

**Connection refused:**
- Check of je IP toegang heeft (Supabase → Settings → Database → Connection Pooling)
- Gebruik connection pooling URL als je buiten Supabase netwerk bent

**Authentication failed:**
- Check of je de juiste password gebruikt
- Service role key is correct

**Table not found:**
- Run het schema.sql script
- Check of tabellen zijn aangemaakt in Supabase dashboard




