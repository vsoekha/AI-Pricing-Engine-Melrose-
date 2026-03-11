"""
Script om demo cases aan te maken voor presentaties.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database_setup import init_database, save_demo_case

# Initialize database
init_database()

# Demo Cases voor Category 1: Bestaand Vastgoed
# Gebaseerd op NL-markt: cap rates logistiek 5,5–7%, kantoor 5–7%, retail 5–7%, resi 3,5–5%.
# NOI en prijzen intern consistent (NOI/cap ≈ waarde, comps in realistische bandbreedte).
demo_cases_category1 = [
    {
        "case_name": "Distributiecentrum Rotterdam Waalhaven",
        "case_type": "existing_asset",
        "category": "category_1",
        "description": "Logistiek distributiecentrum 12.000 m², lease aan grotere huurder, sterke markt en hoge vraag (logistiek ~30% beleggingsvolume NL).",
        "case_data": {
            "asset_type": "logistics",
            "city": "Rotterdam",
            "size_m2": 12000,
            "quality_score": 0.85,
            "noi_annual": 624000,
            "cap_rate_market": 0.062,
            "interest_rate": 0.035,
            "liquidity_index": 0.78,
            "list_price": 9900000,
            "comp_median_price": 10200000
        }
    },
    {
        "case_name": "Kantoor Zuidas Amsterdam",
        "case_type": "existing_asset",
        "category": "category_1",
        "description": "Kantoorpand 5.200 m² in Zuidas, partiële bezetting, uitdagende kantorenmarkt (lage opname 2024).",
        "case_data": {
            "asset_type": "office",
            "city": "Amsterdam",
            "size_m2": 5200,
            "quality_score": 0.72,
            "noi_annual": 325000,
            "cap_rate_market": 0.058,
            "interest_rate": 0.035,
            "liquidity_index": 0.52,
            "list_price": 5850000,
            "comp_median_price": 5600000
        }
    },
    {
        "case_name": "Winkelstraat Retail Eindhoven",
        "case_type": "existing_asset",
        "category": "category_1",
        "description": "Retailpand 1.800 m² aan drukke winkelstraat, gemengde huurders, stabiele cashflow.",
        "case_data": {
            "asset_type": "retail",
            "city": "Eindhoven",
            "size_m2": 1800,
            "quality_score": 0.68,
            "noi_annual": 162000,
            "cap_rate_market": 0.065,
            "interest_rate": 0.035,
            "liquidity_index": 0.62,
            "list_price": 2480000,
            "comp_median_price": 2500000
        }
    },
    {
        "case_name": "Appartementencomplex Utrecht",
        "case_type": "existing_asset",
        "category": "category_1",
        "description": "Multi-family 35 appartementen, totaal 2.800 m², volverhuurd, stabiel rendement (BAR 4,2%).",
        "case_data": {
            "asset_type": "resi",
            "city": "Utrecht",
            "size_m2": 2800,
            "quality_score": 0.76,
            "noi_annual": 298000,
            "cap_rate_market": 0.042,
            "interest_rate": 0.035,
            "liquidity_index": 0.70,
            "list_price": 6950000,
            "comp_median_price": 7100000
        }
    }
]

# Demo Cases voor Category 2: Projectontwikkeling
# Bouwkosten NL 2024: woningen ~€1200–1600/m² (CBS), commercieel/hoger segment €3000–4500/m².
# Verkoopprijzen: Amsterdam €5500–8000/m², Rotterdam €4500–6500/m², Tilburg/Eindhoven lager.
demo_cases_category2 = [
    {
        "case_name": "Woningbouw Amsterdam Sloterdijk",
        "case_type": "development_project",
        "category": "category_2",
        "description": "54 appartementen vrije sector, 4.860 m², build-to-sell. Sterke locatie, realistische bouw-/verkoopcijfers (€3200 bouw, €6200 verkoop).",
        "case_data": {
            "location": "Amsterdam",
            "project_type": "residential",
            "units_count": 54,
            "total_area_m2": 4860,
            "expected_sale_price_per_m2": 6200,
            "build_cost_per_m2": 3200,
            "soft_cost_pct": 0.12,
            "contingency_pct": 0.07,
            "land_cost": 4200000,
            "duration_months": 28,
            "interest_rate": 0.045,
            "target_margin_pct": 0.18
        }
    },
    {
        "case_name": "Mixed-Use Rotterdam Zuid",
        "case_type": "development_project",
        "category": "category_2",
        "description": "Kantoren + retail + 24 wooneenheden, totaal 7.200 m². Gemengd gebruik, hogere soft costs en risico.",
        "case_data": {
            "location": "Rotterdam",
            "project_type": "mixed-use",
            "units_count": 24,
            "total_area_m2": 7200,
            "expected_sale_price_per_m2": 5200,
            "build_cost_per_m2": 3600,
            "soft_cost_pct": 0.14,
            "contingency_pct": 0.08,
            "land_cost": 6500000,
            "duration_months": 32,
            "interest_rate": 0.050,
            "target_margin_pct": 0.16
        }
    },
    {
        "case_name": "Logistiek Hall Tilburg",
        "case_type": "development_project",
        "category": "category_2",
        "description": "Distributiehal 8.000 m², build-to-sell aan belegger. Lager bouwkosten per m², kortere doorlooptijd.",
        "case_data": {
            "location": "Tilburg",
            "project_type": "logistics",
            "units_count": 1,
            "total_area_m2": 8000,
            "expected_sale_price_per_m2": 1450,
            "build_cost_per_m2": 950,
            "soft_cost_pct": 0.10,
            "contingency_pct": 0.06,
            "land_cost": 1200000,
            "duration_months": 14,
            "interest_rate": 0.045,
            "target_margin_pct": 0.15
        }
    },
    {
        "case_name": "Kantoorontwikkeling Den Haag",
        "case_type": "development_project",
        "category": "category_2",
        "description": "Nieuw kantoor 6.000 m², risicovolle kantorenmarkt, hoge bouwkosten en langere doorlooptijd.",
        "case_data": {
            "location": "Den Haag",
            "project_type": "commercial",
            "units_count": 1,
            "total_area_m2": 6000,
            "expected_sale_price_per_m2": 4200,
            "build_cost_per_m2": 3850,
            "soft_cost_pct": 0.16,
            "contingency_pct": 0.09,
            "land_cost": 5800000,
            "duration_months": 36,
            "interest_rate": 0.052,
            "target_margin_pct": 0.12
        }
    }
]


def create_all_demo_cases():
    """Maak alle demo cases aan."""
    all_cases = demo_cases_category1 + demo_cases_category2
    
    created = 0
    for case in all_cases:
        case_id = save_demo_case(
            case["case_name"],
            case["case_type"],
            case["category"],
            case["case_data"],
            case["description"]
        )
        if case_id:
            created += 1
    
    print(f"\n✓ {created} demo cases aangemaakt")
    return created


if __name__ == "__main__":
    print("="*60)
    print("DEMO CASES AANMAKEN")
    print("="*60)
    
    create_all_demo_cases()
    
    print("\nDemo cases zijn klaar voor gebruik!")
    print("Gebruik GET /demo_cases om ze op te halen")

