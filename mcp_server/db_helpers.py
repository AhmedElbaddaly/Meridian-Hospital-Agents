import sqlite3
import os
import re
import sys
import io

if sys.stdout and sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr and sys.stderr.encoding and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ROOT_DIR = os.path.dirname(BASE_DIR)

DB_DIR = os.path.join(ROOT_DIR, "db")

DB_PATH = os.path.join(DB_DIR, "meridian_hospital.db")
SCHEMA_PATH = os.path.join(DB_DIR, "schema.sql")
SEED_PATH = os.path.join(DB_DIR, "seed.sql")

class _AutoCloseConnection(sqlite3.Connection):
    """
    sqlite3.Connection's own `with conn:` block only commits or rolls back
    the transaction -- it never closes the underlying file handle. On
    Linux/macOS a still-open handle doesn't stop a file from being
    deleted/replaced, so this goes unnoticed there; on Windows it does,
    and any later attempt to reset or re-open the database file
    (os.remove, another process opening it exclusively, etc.) fails with
    `PermissionError: [WinError 32] ... used by another process`.

    Subclassing sqlite3.Connection and overriding __exit__ to also close()
    means every existing `with get_connection() as conn:` call site below
    is fixed automatically -- no other line in this file needs to change.
    """

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            self.close()


def get_connection():
    """Establish and return a SQLite database connection with row factory configured.

    BUG FIX (Person 2 / MCP Server Lab correction): the schema.sql defines FOREIGN
    KEY constraints (Admissions -> Patients/Users/Operating_Rooms, ICU_Beds ->
    Patients) but SQLite does NOT enforce foreign keys unless `PRAGMA
    foreign_keys = ON` is issued on every connection. Before this fix, an
    admission or ICU-bed assignment could silently reference a patient_id,
    doctor_id, or room_id that does not exist, and no error was ever raised.
    """
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, factory=_AutoCloseConnection, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL mode so the graph runtime's checkpoint writes and the MCP tool's
    # hospital-data writes can both hit the same physical file concurrently
    # (a state-graph node and an admin-panel action can run at the same time).
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn

def _clean_sql_for_sqlite(sql_content: str) -> str:
    """Dynamically transform MS SQL Server syntax to SQLite dialect."""
    # 1. Remove CREATE DATABASE, USE, and GO commands
    sql_content = re.sub(r'CREATE DATABASE\s+\[?\w+\]?;?', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'USE\s+\[?\w+\]?;?', '', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'^\s*GO\s*$', '', sql_content, flags=re.IGNORECASE | re.MULTILINE)

    # 2. Convert INT IDENTITY(1,1) to INTEGER PRIMARY KEY AUTOINCREMENT
    sql_content = re.sub(r'INT\s+IDENTITY\(\s*1\s*,\s*1\s*\)\s+PRIMARY\s+KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'IDENTITY\(\s*1\s*,\s*1\s*\)', 'AUTOINCREMENT', sql_content, flags=re.IGNORECASE)

    # 3. Convert GETDATE() to CURRENT_TIMESTAMP
    sql_content = re.sub(r'GETDATE\(\)', 'CURRENT_TIMESTAMP', sql_content, flags=re.IGNORECASE)

    # 4. Replace MSSQL specific type definitions
    sql_content = re.sub(r'NVARCHAR\(\w+\)', 'TEXT', sql_content, flags=re.IGNORECASE)
    sql_content = re.sub(r'VARCHAR\(\w+\)', 'TEXT', sql_content, flags=re.IGNORECASE)

    return sql_content

def init_db():
    """Read, sanitize, and execute schema.sql and seed.sql files automatically."""
    print(f"Checking Schema File: {SCHEMA_PATH} -> Exists: {os.path.exists(SCHEMA_PATH)}")
    print(f"Checking Seed File: {SEED_PATH} -> Exists: {os.path.exists(SEED_PATH)}")
    
    with get_connection() as conn:
        cursor = conn.cursor()

        # 1. Execute schema.sql
        if os.path.exists(SCHEMA_PATH):
            with open(SCHEMA_PATH, 'r', encoding='utf-8') as f:
                schema_sql = _clean_sql_for_sqlite(f.read())
                try:
                    cursor.executescript(schema_sql)
                    print("[OK] Schema executed successfully.")
                except Exception as e:
                    if "already exists" in str(e):
                        print("[INFO] Schema tables already exist, skipping execution.")
                    else:
                        print(f"[ERROR] Error executing schema.sql: {e}")
        else:
            print(f"[WARNING] {SCHEMA_PATH} not found!")

        # 2. Execute seed.sql
        if os.path.exists(SEED_PATH):
            with open(SEED_PATH, 'r', encoding='utf-8') as f:
                seed_sql = _clean_sql_for_sqlite(f.read())
                try:
                    cursor.executescript(seed_sql)
                    print("[OK] Seed executed successfully.")
                except Exception as e:
                    if "UNIQUE constraint failed" in str(e) or "already exists" in str(e):
                        print("[INFO] Seed data already initialized, skipping.")
                    else:
                        print(f"[ERROR] Error executing seed.sql: {e}")

        conn.commit()

# Execute automatic database initialization on import
init_db()

# ======================================================
# Database Helper Functions (Matching MCP.py Imports)
# ======================================================

def add_patient(patient_data: dict):
    """Register a new patient into the Patients table."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO Patients (name, age, gender, blood_type, diagnosis)
            VALUES (?, ?, ?, ?, ?)
        """, (patient_data['name'], patient_data['age'], patient_data['gender'], 
              patient_data.get('blood_type'), patient_data.get('diagnosis')))
        conn.commit()
        return cursor.lastrowid

def update_patient_status(patient_id: int, status: str):
    """Update medical triage status for a patient."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE Patients SET status = ? WHERE patient_id = ?", (status, patient_id))
        conn.commit()

def get_patient(patient_id: int):
    """Retrieve patient record by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Patients WHERE patient_id = ?", (patient_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

class HospitalStateError(ValueError):
    """Raised for a hospital-data conflict (double-booking, missing record,
    invalid state transition). Kept as a distinct type so the state-graph
    runtime's failure-ticket path can tell 'unplanned tool failure' apart
    from a plain validation error if it ever wants to."""


def add_admission(admission_data: dict):
    """Create a new hospital admission record, and atomically occupy the
    operating room if one was requested.

    BUG FIX (Person 2): previously this only inserted the Admissions row and
    a *separate* call to update_operating_room_status() was needed to mark
    the room Occupied. Those two writes were never atomic: if the second
    call failed (or was simply never made by the caller), an admission could
    exist against a room the system still believed was 'Available', letting
    a second patient be booked into the same room. Also added an existence
    check for patient_id/doctor_id (FK enforcement alone gives an opaque
    sqlite3.IntegrityError; this gives the ticket system a readable message)
    and a room-availability check to prevent double-booking.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        patient_row = cursor.execute(
            "SELECT patient_id FROM Patients WHERE patient_id = ?",
            (admission_data['patient_id'],),
        ).fetchone()
        if not patient_row:
            raise HospitalStateError(f"No such patient_id={admission_data['patient_id']}")

        doctor_row = cursor.execute(
            "SELECT user_id, role FROM Users WHERE user_id = ?",
            (admission_data['doctor_id'],),
        ).fetchone()
        if not doctor_row:
            raise HospitalStateError(f"No such doctor_id={admission_data['doctor_id']}")

        room_id = admission_data.get('room_id')
        if room_id is not None:
            room_row = cursor.execute(
                "SELECT status FROM Operating_Rooms WHERE room_id = ?", (room_id,)
            ).fetchone()
            if not room_row:
                raise HospitalStateError(f"No such room_id={room_id}")
            if room_row["status"] != "Available":
                raise HospitalStateError(
                    f"Operating Room #{room_id} is '{room_row['status']}', not 'Available' "
                    "-- cannot double-book."
                )

        cursor.execute(
            """
            INSERT INTO Admissions (patient_id, doctor_id, room_id, status)
            VALUES (?, ?, ?, ?)
            """,
            (
                admission_data['patient_id'],
                admission_data['doctor_id'],
                room_id,
                admission_data.get('status', 'Active'),
            ),
        )
        admission_id = cursor.lastrowid

        if room_id is not None:
            cursor.execute(
                "UPDATE Operating_Rooms SET status = 'Occupied' WHERE room_id = ?",
                (room_id,),
            )

        conn.commit()
        return admission_id


def update_room_status(room_id: int, status: str):
    """Update operating room availability status.

    BUG FIX (Person 2): now checks the room exists first, so a bad room_id
    fails with a clear HospitalStateError instead of a silent no-op UPDATE
    (SQLite does not error on an UPDATE that matches zero rows).
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT room_id FROM Operating_Rooms WHERE room_id = ?", (room_id,)
        ).fetchone()
        if not row:
            raise HospitalStateError(f"No such room_id={room_id}")
        cursor.execute("UPDATE Operating_Rooms SET status = ? WHERE room_id = ?", (status, room_id))
        conn.commit()


def update_icu_bed(bed_id: int, patient_id: int = None):
    """Assign an ICU bed to a patient, or release it (patient_id=None).

    BUG FIX (Person 2): previously this blindly overwrote status/patient_id
    with no check of the bed's current state, so two concurrent
    'assign bed #3' calls for two different patients would both succeed --
    the second silently stealing the bed out from under the first patient,
    with no error and no trace. Now the read-check-write happens inside one
    transaction/connection, and assigning an already-Occupied bed to a
    *different* patient raises HospitalStateError. The Hospitals.
    available_icu_beds counter (previously never updated by this function,
    so it drifted from the real ICU_Beds table -- a genuine mapping error)
    is now kept in sync in the same transaction.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        bed_row = cursor.execute(
            "SELECT status, patient_id FROM ICU_Beds WHERE bed_id = ?", (bed_id,)
        ).fetchone()
        if not bed_row:
            raise HospitalStateError(f"No such bed_id={bed_id}")

        if patient_id is not None:
            if bed_row["status"] == "Occupied" and bed_row["patient_id"] != patient_id:
                raise HospitalStateError(
                    f"ICU Bed #{bed_id} is already occupied by patient "
                    f"#{bed_row['patient_id']} -- release it before reassigning."
                )
            patient_row = cursor.execute(
                "SELECT patient_id FROM Patients WHERE patient_id = ?", (patient_id,)
            ).fetchone()
            if not patient_row:
                raise HospitalStateError(f"No such patient_id={patient_id}")

        status = 'Occupied' if patient_id else 'Available'
        was_available = bed_row["status"] == "Available"
        now_available = status == "Available"

        cursor.execute(
            "UPDATE ICU_Beds SET status = ?, patient_id = ? WHERE bed_id = ?",
            (status, patient_id, bed_id),
        )

        # Keep Hospitals.available_icu_beds in sync instead of letting it drift.
        if was_available and not now_available:
            cursor.execute(
                "UPDATE Hospitals SET available_icu_beds = MAX(available_icu_beds - 1, 0)"
            )
        elif now_available and not was_available:
            cursor.execute(
                "UPDATE Hospitals SET available_icu_beds = available_icu_beds + 1"
            )

        conn.commit()


def get_free_icu_beds():
    """Fetch all available ICU beds."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM ICU_Beds WHERE status = 'Available'")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_hospital_info(hospital_id: int = 1):
    """Retrieve hospital capacity details.

    BUG FIX (Person 2): available_icu_beds on the Hospitals row is a
    denormalized counter that can still drift (manual DB edits, seed data,
    a future writer that forgets to touch it). Rather than trust it blindly,
    this now recomputes the true count from ICU_Beds and self-heals the
    stored counter if it disagrees -- so get_hospital_capacity() can never
    report stale bed availability.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT * FROM Hospitals WHERE hospital_id = ?", (hospital_id,)
        ).fetchone()
        if not row:
            return None
        info = dict(row)
        true_count = cursor.execute(
            "SELECT COUNT(*) AS n FROM ICU_Beds WHERE status = 'Available'"
        ).fetchone()["n"]
        if info["available_icu_beds"] != true_count:
            cursor.execute(
                "UPDATE Hospitals SET available_icu_beds = ? WHERE hospital_id = ?",
                (true_count, hospital_id),
            )
            conn.commit()
            info["available_icu_beds"] = true_count
        return info

