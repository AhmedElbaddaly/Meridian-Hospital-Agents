import sys
import os
import asyncio
from typing import Optional, Literal
from pydantic import BaseModel, Field
from mcp.server import MCPServer
import validation
from jsonschema import validate
from schemas import (
    REGISTER_PATIENT_SCHEMA,
    PATIENT_STATUS_SCHEMA,
    ADMISSION_SCHEMA,
    ICU_BED_SCHEMA,
    ROOM_STATUS_SCHEMA
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def log(message):
    print(message, file=sys.stderr)


# ------------------------------------------------------
# 1. Database Helpers Import
# ------------------------------------------------------
try:
    import db_helpers as db
    log("Successfully connected to db_helpers!")
except Exception as e:
    db = None
    log(f"Running in Mock Mode (Error: {e})")

# ------------------------------------------------------
# 2. MCPServer Initialization
# ------------------------------------------------------
mcp = MCPServer("Meridian Hospital Triage MCP Server")

# ------------------------------------------------------
# 3. Pydantic Input Schemas (Validation)
# ------------------------------------------------------


class PatientRegisterInput(BaseModel):
    name: str = Field(..., description="Full name of the patient")
    age: int = Field(..., ge=0, le=120, description="Age of the patient")
    gender: Literal['Male',
                    'Female'] = Field(..., description="Patient gender")
    blood_type: Optional[str] = Field(
        None, description="Blood type e.g. A+, O-, B+")
    diagnosis: Optional[str] = Field(
        None, description="Initial medical diagnosis")


class PatientStatusInput(BaseModel):
    patient_id: int = Field(..., description="ID of the patient")
    status: Literal['Waiting', 'Admitted', 'ICU', 'Surgery',
                    'Discharged'] = Field(..., description="Updated patient status")


class AdmissionInput(BaseModel):
    patient_id: int = Field(...,
                            description="ID of the patient being admitted")
    doctor_id: int = Field(...,
                           description="ID of the assigned doctor (Users table)")
    room_id: Optional[int] = Field(
        None, description="Optional operating room ID")
    status: Literal['Active', 'Completed', 'Transferred'] = 'Active'


class ICUBedAssignmentInput(BaseModel):
    bed_id: int = Field(..., description="ICU Bed ID")
    patient_id: Optional[int] = Field(
        None, description="Patient ID to assign, or None to release bed")


class RoomStatusInput(BaseModel):
    room_id: int = Field(..., description="Operating Room ID")
    status: Literal['Available', 'Occupied',
                    'Maintenance'] = Field(..., description="Operating room status")

# ------------------------------------------------------
# ------------------------------------------------------
# 4. MCP Tools (Database Operations)
# ------------------------------------------------------

# --- Patients Tools ---


@mcp.tool()
def register_patient(patient: PatientRegisterInput) -> str:
    """Register a new patient into the Meridian Hospital database."""

    validation.authorize("admin", "register_patient")

    payload = patient.model_dump()

    validate(
        instance=payload,
        schema=REGISTER_PATIENT_SCHEMA
    )

    validation.validate_patient_registration(payload)

    if db and hasattr(db, "add_patient"):
        res = db.add_patient(payload)
        return f"Patient registered successfully in database: {res}"

    return (
        f"Mock: Registered patient '{patient.name}' "
        f"(Age: {patient.age}, Gender: {patient.gender}) with ID #101."
    )


@mcp.tool()
def update_patient_status(data: PatientStatusInput) -> str:
    """Update medical and triage status of an existing patient."""

    validation.authorize("doctor", "update_patient_status")

    payload = data.model_dump()

    validate(
        instance=payload,
        schema=PATIENT_STATUS_SCHEMA
    )

    validation.validate_patient_status(data.status)

    if db and hasattr(db, "update_patient_status"):
        db.update_patient_status(data.patient_id, data.status)
        return (
            f"Patient #{data.patient_id} status successfully "
            f"updated to '{data.status}'."
        )

    return f"Mock: Patient #{data.patient_id} status updated to '{data.status}'."


@mcp.tool()
def get_patient_details(patient_id: int) -> str:
    """Retrieve full details for a specific patient by ID."""

    if patient_id <= 0:
        raise ValueError("Invalid patient ID.")

    if db and hasattr(db, "get_patient"):
        patient = db.get_patient(patient_id)
        return f"Patient Details: {patient}"

    return (
        f"Mock Details for Patient #{patient_id}: "
        "Name: John Doe, Age: 45, Gender: Male, Status: Waiting."
    )


# --- Admissions & Operating Rooms Tools ---
@mcp.tool()
def create_admission(admission: AdmissionInput) -> str:
    """Create a new admission record linking patient, doctor, and optional operating room."""

    validation.authorize("doctor", "create_admission")

    payload = admission.model_dump()

    validate(
        instance=payload,
        schema=ADMISSION_SCHEMA
    )

    validation.validate_admission(
        admission.patient_id,
        admission.doctor_id
    )

    if db and hasattr(db, "add_admission"):
        res = db.add_admission(payload)
        return f"Admission created successfully: {res}"

    return (
        f"Mock: Admission created for Patient "
        f"#{admission.patient_id} assigned to Doctor #{admission.doctor_id}."
    )


@mcp.tool()
def update_operating_room_status(data: RoomStatusInput) -> str:
    """Update status of an operating room."""

    validation.authorize("admin", "update_operating_room_status")

    payload = data.model_dump()

    validate(
        instance=payload,
        schema=ROOM_STATUS_SCHEMA
    )

    validation.validate_room_status(
        data.room_id,
        data.status
    )

    if db and hasattr(db, "update_room_status"):
        db.update_room_status(data.room_id, data.status)
        return (
            f"Operating Room #{data.room_id} "
            f"status updated to '{data.status}'."
        )

    return f"Mock: Operating Room #{data.room_id} set to '{data.status}'."


# --- ICU Beds Tools ---
@mcp.tool()
def manage_icu_bed(data: ICUBedAssignmentInput) -> str:
    """Assign or release an ICU bed."""

    validation.authorize("doctor", "manage_icu_bed")

    payload = data.model_dump()

    validate(
        instance=payload,
        schema=ICU_BED_SCHEMA
    )

    validation.validate_icu_assignment(
        data.bed_id,
        data.patient_id
    )

    if db and hasattr(db, "update_icu_bed"):
        db.update_icu_bed(data.bed_id, data.patient_id)

        action = (
            f"assigned to Patient #{data.patient_id}"
            if data.patient_id
            else "released"
        )

        return f"ICU Bed #{data.bed_id} successfully {action}."

    action = (
        f"assigned to Patient #{data.patient_id}"
        if data.patient_id
        else "released"
    )

    return f"Mock: ICU Bed #{data.bed_id} successfully {action}."


@mcp.tool()
def get_available_icu_beds() -> str:
    """Fetch all available ICU beds."""

    if db and hasattr(db, "get_free_icu_beds"):
        beds = db.get_free_icu_beds()
        return f"Available ICU Beds: {beds}"

    return (
        "Mock: Available ICU Beds -> "
        "[Bed #1 (ICU-101), Bed #3 (ICU-103), Bed #5 (ICU-105)]"
    )


# --- Hospitals & Medical Staff Tools ---
@mcp.tool()
def get_hospital_capacity(hospital_id: int = 1) -> str:
    """Check hospital capacity."""

    if hospital_id <= 0:
        raise ValueError("Invalid hospital ID.")

    if db and hasattr(db, "get_hospital_info"):
        info = db.get_hospital_info(hospital_id)
        return f"Hospital Info: {info}"

    return "Mock: Meridian General Hospital | City: Central | Available ICU Beds: 8"
# ------------------------------------------------------
# 5. MCP Resources (Read-Only Context)
# ------------------------------------------------------


@mcp.resource("triage://protocols/guidelines")
def get_triage_guidelines() -> str:
    """Returns the emergency triage protocol guidelines for clinical assessment."""
    return """
    =====================================================
    MERIDIAN HOSPITAL EMERGENCY TRIAGE GUIDELINES
    =====================================================
    1. RED LEVEL (Critical / Life-Threatening):
       - Conditions: Cardiac arrest, severe trauma, respiratory failure.
       - Action: Immediate assignment to ICU Bed or Operating Room. Set status to 'ICU' or 'Surgery'.
    
    2. YELLOW LEVEL (Urgent):
       - Conditions: Severe asthma, acute abdominal pain, high fever.
       - Action: Admit patient and assign attending Doctor. Set status to 'Admitted'.
    
    3. GREEN LEVEL (Non-Urgent):
       - Conditions: Minor lacerations, sprains, mild symptoms.
       - Action: Register patient, set status to 'Waiting'.
    """


@mcp.resource("hospital://operating-rooms/rules")
def get_or_rules() -> str:
    """Returns protocol rules for operating room utilization."""
    return """
    OPERATING ROOM (OR) PROTOCOLS:
    - Rooms must be marked 'Maintenance' immediately after surgical procedures.
    - Status can only be changed to 'Available' after full sanitation verification.
    """

# ------------------------------------------------------
# 6. MCP Prompts (AI Agent Action Templates)
# ------------------------------------------------------


@mcp.prompt()
def triage_patient_prompt(patient_name: str, age: int, symptoms: str) -> str:
    """Template prompt to guide AI Agent in assessing patient urgency and executing tools."""
    return f"""
    You are an Emergency Triage AI Assistant for Meridian Hospital.
    
    Patient Evaluation Request:
    - Name: {patient_name}
    - Age: {age}
    - Reported Symptoms: "{symptoms}"
    
    Instructions:
    1. Check 'triage://protocols/guidelines' resource to classify urgency level.
    2. Register the patient using `register_patient`.
    3. If critical, assign an available ICU bed using `manage_icu_bed` and update patient status to 'ICU'.
    4. If non-critical, assign status 'Waiting' or 'Admitted'.
    """


# ------------------------------------------------------
# 7. Execution Entry Point
# ------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    asyncio.run(mcp.run())
