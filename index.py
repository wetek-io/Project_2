# Direct Indicators of Heart Disease
direct_indicators = ["HadHeartAttack", "HadAngina", "HadStroke"]

# Indirect/Contributing Indicators of Heart Disease
indirect_indicators = [
    "Sex",
    "GeneralHealth",
    "PhysicalHealthDays",
    "MentalHealthDays",
    "PhysicalActivities",
    "SleepHours",
    "RemovedTeeth",
    "HadCOPD",
    "HadDepressiveDisorder",
    "HadDiabetes",
    "SmokerStatus",
    "AlcoholDrinkers",
    "AgeCategory",
    "BMI",
    "WeightInKilograms",
]

# Less Direct but Relevant Indicators
less_direct_indicators = ["RaceEthnicityCategory", "HighRiskLastYear", "HIVTesting"]

# Optional or Less Likely Indicators
optional_indicators = [
    "CovidPos",
    "FluVaxLast12",
    "PneumoVaxEver",
    "TetanusLast10Tdap",
    "ChestScan",
]
