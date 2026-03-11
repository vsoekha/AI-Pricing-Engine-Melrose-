"""
Script om CSV data te converteren naar Nederlandse kolomnamen voor presentaties.
Dit is alleen voor display doeleinden - training gebruikt altijd Engelse kolommen.
"""

import pandas as pd
from pathlib import Path

# Mapping van Engelse naar Nederlandse kolomnamen
COLUMN_MAPPING = {
    'asset_type': 'Asset Type',
    'city': 'Stad',
    'size_m2': 'Oppervlakte (m²)',
    'quality_score': 'Kwaliteitsscore',
    'noi_annual': 'NOI Jaarlijks (€)',
    'cap_rate_market': 'Cap Rate Markt',
    'interest_rate': 'Rentestand',
    'liquidity_index': 'Liquiditeitsindex',
    'list_price': 'Vraagprijs (€)',
    'comp_median_price': 'Mediaan Prijs Vergelijkbaar (€)',
    'sold_within_180d': 'Verkocht Binnen 180 Dagen'
}

# Asset type vertaling
ASSET_TYPE_MAPPING = {
    'logistics': 'Logistiek',
    'office': 'Kantoor',
    'resi': 'Residentieel',
    'retail': 'Retail',
    'mixed': 'Gemengd'
}

# Sold vertaling
SOLD_MAPPING = {
    1: 'Ja',
    0: 'Nee'
}


def convert_csv_to_dutch(input_path: str, output_path: str):
    """
    Converteer CSV met Engelse kolommen naar Nederlandse kolommen voor presentatie.
    
    Args:
        input_path: Pad naar originele CSV (Engelse kolommen)
        output_path: Pad naar output CSV (Nederlandse kolommen)
    """
    print(f"Laden data van {input_path}...")
    
    # Detect delimiter
    with open(input_path, 'r') as f:
        first_line = f.readline()
        delimiter = ';' if ';' in first_line else ','
    
    # Load data
    df = pd.read_csv(input_path, delimiter=delimiter)
    
    print(f"Geladen: {len(df)} rijen, {len(df.columns)} kolommen")
    
    # Rename columns
    df_renamed = df.rename(columns=COLUMN_MAPPING)
    
    # Translate asset_type
    if 'Asset Type' in df_renamed.columns:
        df_renamed['Asset Type'] = df_renamed['Asset Type'].map(ASSET_TYPE_MAPPING)
    
    # Translate sold_within_180d
    if 'Verkocht Binnen 180 Dagen' in df_renamed.columns:
        df_renamed['Verkocht Binnen 180 Dagen'] = df_renamed['Verkocht Binnen 180 Dagen'].map(SOLD_MAPPING)
    
    # Format numbers for display
    if 'NOI Jaarlijks (€)' in df_renamed.columns:
        df_renamed['NOI Jaarlijks (€)'] = df_renamed['NOI Jaarlijks (€)'].apply(lambda x: f"{x:,.0f}".replace(',', '.'))
    
    if 'Vraagprijs (€)' in df_renamed.columns:
        df_renamed['Vraagprijs (€)'] = df_renamed['Vraagprijs (€)'].apply(lambda x: f"{x:,.0f}".replace(',', '.'))
    
    if 'Mediaan Prijs Vergelijkbaar (€)' in df_renamed.columns:
        df_renamed['Mediaan Prijs Vergelijkbaar (€)'] = df_renamed['Mediaan Prijs Vergelijkbaar (€)'].apply(lambda x: f"{x:,.0f}".replace(',', '.'))
    
    # Format percentages
    if 'Cap Rate Markt' in df_renamed.columns:
        df_renamed['Cap Rate Markt'] = df_renamed['Cap Rate Markt'].apply(lambda x: f"{x*100:.2f}%")
    
    if 'Rentestand' in df_renamed.columns:
        df_renamed['Rentestand'] = df_renamed['Rentestand'].apply(lambda x: f"{x*100:.2f}%")
    
    if 'Kwaliteitsscore' in df_renamed.columns:
        df_renamed['Kwaliteitsscore'] = df_renamed['Kwaliteitsscore'].apply(lambda x: f"{x*100:.0f}%")
    
    if 'Liquiditeitsindex' in df_renamed.columns:
        df_renamed['Liquiditeitsindex'] = df_renamed['Liquiditeitsindex'].apply(lambda x: f"{x*100:.0f}%")
    
    # Save with semicolon delimiter (Dutch Excel standard)
    df_renamed.to_csv(output_path, sep=';', index=False)
    
    print(f"✓ Opgeslagen naar {output_path}")
    print(f"  {len(df_renamed)} rijen, {len(df_renamed.columns)} kolommen")
    print("\nLet op: Deze versie is alleen voor presentatie!")
    print("Voor training gebruik altijd de originele listings.csv")


if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"
    input_file = data_dir / "listings.csv"
    output_file = data_dir / "listings_nl_presentatie.csv"
    
    if not input_file.exists():
        print(f"Fout: {input_file} niet gevonden!")
        exit(1)
    
    convert_csv_to_dutch(str(input_file), str(output_file))
    print("\n✓ Conversie compleet!")




