"""
Validation and authorization helpers for MCP tool requests.
"""

from typing import Optional

# Allowed Values
# ------------------------------------------------------
VALID_BLOOD_TYPES = {
    "A+", "A-", "B+", "B-",
    "AB+", "AB-", "O+", "O-"
}

PATIENT_STATUSES = {
    "Waiting",
    "Admitted",
    "ICU",
    "Surgery",
    "Discharged"
}

ROOM_STATUSES = {
    "Available",
    "Occupied",
    "Maintenance"
}

# Patient Validation
# ------------------------------------------------------
def validate_patient_registration(data: dict):
    name = data.get("name", "").strip()
    if not name:
        raise ValueError("Patient name cannot be empty.")

    age = data.get("age")
    if age is None or age < 0 or age > 100:
        raise ValueError("Age must be between 0 and 120.")

    gender = data.get("gender")
    if gender not in {"Male", "Female"}:
        raise ValueError("Invalid gender.")

    blood_type = data.get("blood_type")
    if blood_type and blood_type not in VALID_BLOOD_TYPES:
        raise ValueError("Invalid blood type.")
    
# Status Validation
# ------------------------------------------------------
def validate_patient_status(status: str):
    if status not in PATIENT_STATUSES:
        raise ValueError("Invalid patient status.")
    
# Admission Validation
# ------------------------------------------------------

def validate_admission(patient_id: int, doctor_id: int):
    if patient_id <= 0:
        raise ValueError("Invalid patient ID.")

    if doctor_id <= 0:
        raise ValueError("Invalid doctor ID.")
    
# ICU Validation
# ------------------------------------------------------
def validate_icu_assignment(bed_id: int, patient_id: Optional[int]):
    if bed_id <= 0:
        raise ValueError("Invalid ICU bed ID.")

    if patient_id is not None and patient_id <= 0:
        raise ValueError("Invalid patient ID.")
    
# Operating Room Validation
# ------------------------------------------------------

def validate_room_status(room_id: int, status: str):
    if room_id <= 0:
        raise ValueError("Invalid operating room ID.")

    if status not in ROOM_STATUSES:
        raise ValueError("Invalid operating room status.")

# Authorization
# ------------------------------------------------------
PERMISSIONS = {
    "admin": {
        "register_patient",
        "update_patient_status",
        "create_admission",
        "manage_icu_bed",
        "update_operating_room_status"
    },
    "doctor": {
        "update_patient_status",
        "create_admission",
        "manage_icu_bed"
    },
    "nurse": {
        "update_patient_status"
    }
}


def authorize(user_role: str, action: str):
    allowed_actions = PERMISSIONS.get(user_role, set())

    if action not in allowed_actions:
        raise PermissionError(
            f"User role '{user_role}' is not allowed to perform '{action}'."
        )