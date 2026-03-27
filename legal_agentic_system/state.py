from typing import TypedDict, Optional


class CitizenState(TypedDict):
    # --- Legal Dashboard Inputs ---
    file_path: Optional[str]
    raw_text: Optional[str]
    legal_category: Optional[str]
    case_status: Optional[str]

    # --- Profile Dashboard Inputs ---
    user_age: Optional[str]
    user_state: Optional[str]
    user_education: Optional[str]
    user_specialization: Optional[str]
    user_gender: Optional[str]
    annual_income: Optional[str]
    current_status: Optional[str]
    extracurriculars: Optional[str]
    user_goal: Optional[str]

    # --- Legal Outputs ---
    legal_summary: str
    legal_solution: str
    gov_resources: str

    # --- Profile Outputs ---
    active_opportunities: str
    missed_opportunities: str
    opportunities_json: list[dict]
    missed_opportunities_json: list[dict]
    guiding_roadmap: str
    insight_report: str


# Backward compatibility for existing modules still importing LegalGraphState
LegalGraphState = CitizenState