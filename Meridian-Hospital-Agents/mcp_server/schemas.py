"""
Contains the JSON schemas for all tool inputs,
including required fields, allowed values,
and validation constraints.
"""
REGISTER_PATIENT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Full name of the patient"
        },
        "age": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Age of the patient"
        },
        "gender": {
            "type": "string",
            "enum": ["Male", "Female"],
            "description": "Patient gender"
        },
        "blood_type": {
            "type": ["string", "null"],
            "description": "Blood type (e.g. A+, O-, B+)"
        },
        "diagnosis": {
            "type": ["string", "null"],
            "description": "Initial medical diagnosis"
        }
    },
    "required": ["name", "age", "gender"],
    "additionalProperties": False
}


PATIENT_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "patient_id": {
            "type": "integer",
            "minimum": 1,
            "description": "Patient ID"
        },
        "status": {
            "type": "string",
            "enum": [
                "Waiting",
                "Admitted",
                "ICU",
                "Surgery",
                "Discharged"
            ],
            "description": "Updated patient status"
        }
    },
    "required": ["patient_id", "status"],
    "additionalProperties": False
}


ADMISSION_SCHEMA = {
    "type": "object",
    "properties": {
        "patient_id": {
            "type": "integer",
            "minimum": 1,
            "description": "Patient ID"
        },
        "doctor_id": {
            "type": "integer",
            "minimum": 1,
            "description": "Assigned doctor ID"
        },
        "room_id": {
            "type": ["integer", "null"],
            "description": "Operating room ID"
        },
        "status": {
            "type": "string",
            "enum": [
                "Active",
                "Completed",
                "Transferred"
            ],
            "description": "Admission status"
        }
    },
    "required": [
        "patient_id",
        "doctor_id"
    ],
    "additionalProperties": False
}


ICU_BED_SCHEMA = {
    "type": "object",
    "properties": {
        "bed_id": {
            "type": "integer",
            "minimum": 1,
            "description": "ICU Bed ID"
        },
        "patient_id": {
            "type": ["integer", "null"],
            "description": "Patient ID assigned to the bed"
        }
    },
    "required": [
        "bed_id"
    ],
    "additionalProperties": False
}


ROOM_STATUS_SCHEMA = {
    "type": "object",
    "properties": {
        "room_id": {
            "type": "integer",
            "minimum": 1,
            "description": "Operating room ID"
        },
        "status": {
            "type": "string",
            "enum": [
                "Available",
                "Occupied",
                "Maintenance"
            ],
            "description": "Operating room status"
        }
    },
    "required": [
        "room_id",
        "status"
    ],
    "additionalProperties": False
}