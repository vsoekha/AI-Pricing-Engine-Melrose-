# Base44 Frontend Update - Category 2 Development Projects

## API Contract

### New Endpoint: Development Project Analysis

**Endpoint:** `POST https://ai-pricing-engine-melrose.onrender.com/dev_project_analysis`

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
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
  "target_margin_pct": 0.18,
  "exit_cap_rate": null,
  "absorption_months": null
}
```

**Response:**
```json
{
  "project_summary": {
    "location": "Amsterdam",
    "project_type": "residential",
    "units_count": 50,
    "total_area_m2": 5000,
    "duration_months": 24,
    "target_margin_pct": 0.18
  },
  "base_scenario": {
    "scenario_name": "Base Case",
    "total_revenue": 30000000.0,
    "total_project_cost": 24650000.0,
    "profit": 5350000.0,
    "margin": 0.1783,
    "meets_target_margin": false
  },
  "downside_scenario": {
    "scenario_name": "Downside (-7% price, +7% cost, +0.5% rate)",
    "total_revenue": 27900000.0,
    "total_project_cost": 26475500.0,
    "profit": 1424500.0,
    "margin": 0.0511,
    "meets_target_margin": false
  },
  "upside_scenario": {
    "scenario_name": "Upside (+5% price, -3% cost, -0.5% rate)",
    "total_revenue": 31500000.0,
    "total_project_cost": 23340500.0,
    "profit": 8159500.0,
    "margin": 0.2590,
    "meets_target_margin": true
  },
  "recommended_action": "REVIEW",
  "recommendation_reason": "Base case is profitable but downside scenario is risky - review assumptions"
}
```

---

## Base44 Frontend Update Prompt

```
Update the frontend to support two categories of real estate analysis:

CATEGORY 1: Existing Assets (already implemented)
- Endpoint: POST /recommend_price
- Current form and results display (keep unchanged)

CATEGORY 2: Development Projects (NEW - to be added)
- Endpoint: POST /dev_project_analysis
- New form and results display

UI CHANGES REQUIRED:

1. ADD CATEGORY SELECTOR:
   - Add tabs or toggle at the top: "Existing Asset" | "Development Project"
   - Default to "Existing Asset" (current functionality)
   - When "Development Project" is selected, show new form

2. DEVELOPMENT PROJECT FORM:
   Create a new form with these fields:
   
   Required Fields:
   - location (text input): City/location name
   - project_type (dropdown/select): residential, mixed-use, commercial, office, logistics
   - units_count (number input): Number of units
   - total_area_m2 (number input): Total development area in m²
   - expected_sale_price_per_m2 (number input): Expected sale price per m² (€)
   - build_cost_per_m2 (number input): Build cost per m² (€)
   - soft_cost_pct (number input): Soft costs percentage (e.g., 12 for 12%)
   - contingency_pct (number input): Contingency percentage (e.g., 7 for 7%)
   - land_cost (number input): Land acquisition cost (€)
   - duration_months (number input): Project duration in months
   - interest_rate (number input): Financing interest rate (e.g., 4.5 for 4.5%)
   - target_margin_pct (number input): Target profit margin (e.g., 18 for 18%)
   
   Optional Fields:
   - exit_cap_rate (number input, optional): Exit cap rate
   - absorption_months (number input, optional): Expected absorption period

3. RESULTS DISPLAY FOR DEVELOPMENT PROJECTS:
   When form is submitted, display results in organized sections:
   
   a) PROJECT SUMMARY CARD:
      - Location, Project Type, Units, Total Area, Duration, Target Margin
   
   b) SCENARIOS SECTION:
      Display 3 scenario cards side-by-side (or stacked on mobile):
      
      BASE CASE:
      - Total Revenue: €30,000,000 (formatted with commas)
      - Total Project Cost: €24,650,000
      - Profit: €5,350,000
      - Margin: 17.83% (show as percentage)
      - Meets Target: ✓ or ✗ (with color: green/red)
      
      DOWNSIDE SCENARIO:
      - Same fields as Base Case
      - Label: "Downside (-7% price, +7% cost, +0.5% rate)"
      - Use warning/orange color scheme
      
      UPSIDE SCENARIO:
      - Same fields as Base Case
      - Label: "Upside (+5% price, -3% cost, -0.5% rate)"
      - Use success/green color scheme
   
   c) RECOMMENDATION CARD:
      - Recommended Action: Large badge/button showing "PROCEED", "REVIEW", "REPRICE", or "WAIT"
      - Color code:
        * PROCEED: Green
        * REVIEW: Yellow/Orange
        * REPRICE: Orange
        * WAIT: Red
      - Recommendation Reason: Text explanation below the action
   
   d) VISUAL INDICATORS:
      - Profit bars/charts comparing scenarios
      - Margin percentage with visual gauge
      - Color-coded meets_target_margin indicators

4. FORM VALIDATION:
   - All required fields must be filled
   - Numbers must be positive
   - Percentages should be between 0-100 (convert to decimal: divide by 100 before sending)
   - Interest rate should be between 0-100 (convert to decimal: divide by 100)
   - Show validation errors clearly

5. API INTEGRATION:
   - When "Development Project" form is submitted:
     * Convert percentages: soft_cost_pct/100, contingency_pct/100, interest_rate/100, target_margin_pct/100
     * POST to: https://ai-pricing-engine-melrose.onrender.com/dev_project_analysis
     * Handle loading state
     * Display results in the format described above
     * Handle errors gracefully

6. RESPONSIVE DESIGN:
   - Tabs work on mobile
   - Scenario cards stack on mobile
   - Form is mobile-friendly
   - Results are readable on all screen sizes

7. USER EXPERIENCE:
   - Clear visual distinction between the two categories
   - Smooth transition when switching categories
   - Loading indicators during API calls
   - Error messages if API fails
   - Success feedback when analysis completes

IMPLEMENTATION NOTES:
- Keep existing "Existing Asset" functionality completely unchanged
- Add new functionality alongside, not replacing
- Use consistent styling with existing design
- Ensure CORS is configured (already done in backend)
- Test both categories work independently

EXAMPLE API CALL (JavaScript):
```javascript
// When Development Project form is submitted
const response = await fetch('https://ai-pricing-engine-melrose.onrender.com/dev_project_analysis', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    location: formData.location,
    project_type: formData.project_type,
    units_count: parseInt(formData.units_count),
    total_area_m2: parseFloat(formData.total_area_m2),
    expected_sale_price_per_m2: parseFloat(formData.expected_sale_price_per_m2),
    build_cost_per_m2: parseFloat(formData.build_cost_per_m2),
    soft_cost_pct: parseFloat(formData.soft_cost_pct) / 100,  // Convert to decimal
    contingency_pct: parseFloat(formData.contingency_pct) / 100,
    land_cost: parseFloat(formData.land_cost),
    duration_months: parseInt(formData.duration_months),
    interest_rate: parseFloat(formData.interest_rate) / 100,
    target_margin_pct: parseFloat(formData.target_margin_pct) / 100,
    exit_cap_rate: formData.exit_cap_rate ? parseFloat(formData.exit_cap_rate) / 100 : null,
    absorption_months: formData.absorption_months ? parseInt(formData.absorption_months) : null
  })
});

const result = await response.json();
// Display result in UI
```

```





