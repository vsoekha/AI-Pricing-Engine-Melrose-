# Datastromen: Demo vs Live – en hoe het in de demo zichtbaar te maken

## Doel
Tijdens de demo moet duidelijk zijn: **de prijs wordt (in de live situatie) gevoed door realtime data, stats en externe bronnen**. De techniek/koppeling wil je al in de demo tonen, omdat dat de echte waarde laat zien.

---

## Wat de demo NU gebruikt (alleen gebruikersinput)

| Input in demo | In jouw documentatie | Live situatie (waar het vandaan komt) |
|---------------|----------------------|----------------------------------------|
| **Cap rate** (gebruiker vult in) | Yield & Cap Rate Data | Cap rate per asset type/stad uit marktrapporten (CBRE, JLL, CoStar, BAR-surveys) |
| **Rente** (gebruiker vult in) | Macro & Interest Rate Data | ECB, 10y treasury, hypotheekrente – realtime/feed |
| **NOI** (gebruiker vult in) | Asset Performance Data (NOI) | Exploitatieoverzicht, T-12, rent roll |
| **Vraagprijs / comp_median_price** (gebruiker vult in) | Transaction Data (comps) | Comps, NVM Business, RealNext, broker reports |
| **Liquiditeitsindex** (gebruiker vult in) | Market Liquidity Data | Transactievolume, opname, kwartaalrapporten |
| **Asset type, stad, oppervlakte** | Basis asset info | Dossier, OVL, kadaster |
| **Kwaliteitsscore** | Afgeleid van dossier/renovatie | Conditie, energielabel, huurderskwaliteit |

Conclusie: **in de demo zijn dit allemaal “handmatige” velden. In live situatie zouden veel van deze velden automatisch gevuld worden uit jouw datastromen.**

---

## Wat je in de demo kunt doen: “de koppeling zichtbaar maken”

Je wilt niet per se nu al alle APIs bouwen, maar wél dat de **techniek en koppeling** zichtbaar zijn. Drie lagen:

### 1. UI: “Data-bronnen” bij het resultaat (aanbevolen)

In het resultatenscherm een klein blok toevoegen, bijvoorbeeld:

**“Waarop is dit advies gebaseerd?”**

- Cap rate → *In live: marktdata (bijv. CBRE/JLL cap rate survey). Nu: uw invoer.*  
- Rente → *In live: ECB / rente-feed. Nu: uw invoer.*  
- Comps / vergelijkbare prijs → *In live: transactiedata (NVM Business, RealNext). Nu: uw invoer.*  
- Marktliquiditeit → *In live: transactievolume, opname. Nu: uw inschatting (liquiditeitsindex).*  
- Verkoopkans (AI) → *Getraind op historische transacties; in live aangevuld met actuele comps en marktdata.*

Zo maak je in de demo expliciet: **dezelfde velden die u nu invult, komen in de live versie uit jullie datastromen.**

### 2. Eén echte realtime koppeling (sterk voor de demo)

Eén duidelijke “live” bron in de demo geeft veel geloofwaardigheid, bijvoorbeeld:

- **Rente:**  
  - Optioneel: bij laden van het formulier de actuele ECB-rente (of een andere publieke rente-API) ophalen en het veld “Rente” voorinvullen.  
  - In de UI: “Rente: 3,50% (bron: ECB, actueel)” vs “Rente: [handmatig aangepast]”.

Als je wilt, kunnen we in een volgende stap concreet één zo’n bron (bijv. ECB of een gratis rente-API) uitzoeken en in de backend/ frontend aansluiten.

### 3. Eén A4 “Live-architectuur” voor onder de demo

Een simpel plaatje of A4 dat je bij de demo laat zien:

- Links: **Jouw datastromen** (zoals in je documentatie):  
  Asset performance, transactiedata, cap rate, liquiditeit, rente, buyer demand, (voor development: bouwkosten, pro forma, absorptie, etc.)
- Midden: **Pricing Engine** (wat de demo toont):  
  Zelfde logica: optimale prijs, timing, scenario’s.
- Rechts: **Output:**  
  Adviesprijs, verkoopkans, timing, scenario’s.

Tekst erbij: *“In de demo vult u een aantal velden handmatig in. In de live omgeving worden deze velden gevoed door de getoonde datastromen (real-time of dagelijkse updates).”*

Dan is de boodschap helder: **de techniek en koppeling zijn er; de demo gebruikt dezelfde “ingangen” die in live uit jullie bronnen komen.**

---

## Jouw documentatie gekoppeld aan de demo

### Categorie 1 – Bestaand vastgoed (demo)

- **Scenario’s die je noemt:** Cashflow machine, Value-add, Opportunistisch (market timing).  
- **Demo:** Eén flow; het model geeft o.a. optimale prijs, timing (Q1/Q2/Q3), rente-scenario’s, kopersprofiel.  
- **Koppeling:**  
  - “Cashflow machine” → NOI, cap rate, comps (in live: exploitatie + transactiedata).  
  - “Value-add” →zelfde engine, andere input (na renovatie); in live: NOI-prognose + comps.  
  - “Opportunistisch” → timing + rente-scenario’s; in live: rente-feed + liquiditeitsdata.

Datastromen die in live de demo-velden voeden:

- Asset Performance Data → NOI, (impliciet kwaliteit).  
- Transaction Data (comps) → vergelijkbare prijs, cap rate.  
- Yield & Cap Rate Data → cap rate.  
- Market Liquidity Data → liquiditeitsindex.  
- Macro & Interest Rate Data → rente.  
- Buyer Demand Data → kopersprofiel (nu model; in live eventueel aangevuld met marktrapporten).

### Categorie 2 – Projectontwikkeling (demo)

- **Scenario’s:** Build-to-sell, Build-to-rent, Value-add during development.  
- **Demo:** Basis / negatief / positief scenario, doelmarge, aanbeveling (DOORGAAN/HERZIEN/etc.).  
- **Koppeling:**  
  - Development Cost Data → bouwkosten, soft costs, grondkosten (in live: begrotingen, stichtingskosten).  
  - Pro forma / forecasted income → in live: pro forma + exploitatieprognose.  
  - New build sales comps → verkoopprijs/m² (in live: vergelijkbare nieuwbouw).  
  - Macro, rente, demografie → rente, timing (in live: ECB, bouwkostenindex, woonvisies).

Je kunt in de demo bij “Projectontwikkeling” dezelfde “Data-bronnen”-tekst toevoegen: *“Verkoopprijs/m², bouwkosten, rente: in live uit [comps, stichtingskosten, ECB/feed].”*

---

## Korte boodschap voor tijdens de demo

- “Wat u hier ziet, is dezelfde engine die we in de live omgeving gebruiken. Het verschil: nu vult u een aantal velden zelf in; bij jullie koppelen we **transactiedata, cap rates, rente en marktliquiditeit** uit jullie bronnen (CBRE, NVM, ECB, etc.), zodat het advies op actuele data gebaseerd is.”
- “De waarde zit dus in twee dingen: **de reken- en AI-logica** die u hier ziet, én de **koppeling met jullie datastromen** zodat die velden automatisch gevoed worden.”

---

## Volgende stap (concreet)

1. **Direct in de demo:**  
   In de frontend (Base44) een klein blok “Waarop is dit advies gebaseerd?” toevoegen met de mapping: veld → “Nu: uw invoer” / “In live: [bron]”.  
   Optioneel: één echte koppeling (bijv. rente) zodat je kunt zeggen: “De rente haalten we hier al live op.”

2. **Voor jezelf en voor klanten:**  
   Eén A4 “Live-architectuur” (datastromen → engine → advies) gebruiken bij elke demo.

Als je wilt, kan ik in een volgende stap:
- de exacte teksten voor dat “Data-bronnen”-blok in het Nederlands uitschrijven (per veld), en/of  
- een voorstel doen voor één rente-API en waar in de code die zou aansluiten (zonder alles al te bouwen).

Dit document kun je ook als basis gebruiken voor je eigen demo-script en voor gesprekken met potentiële klanten (en voor KVK/positioning: “AI pricing op basis van realtime markt- en transactiedata”).
