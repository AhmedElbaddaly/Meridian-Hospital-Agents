"""
Person 2 deliverable, part 3: standalone proof that the specific
mcp_server/db_helpers bugs are actually fixed (not just described in a
comment). Run directly against the real seeded hospital DB.

Run:
    python -m state_graph.demo_person2_db_fixes
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp_server"))
import db_helpers as db  # noqa: E402


def check(label, fn):
    try:
        fn()
        print(f"[FAIL] {label}: expected an error, none was raised")
        return False
    except db.HospitalStateError as e:
        print(f"[PASS] {label}: correctly rejected -> {e}")
        return True
    except Exception as e:
        print(f"[FAIL] {label}: wrong exception type {type(e).__name__}: {e}")
        return False


def main():
    ok = True

    print("--- Foreign key enforcement ---")
    ok &= check(
        "admission with nonexistent patient_id",
        lambda: db.add_admission({"patient_id": 999999, "doctor_id": 1}),
    )
    ok &= check(
        "admission with nonexistent doctor_id",
        lambda: db.add_admission({"patient_id": 1, "doctor_id": 999999}),
    )

    print("\n--- Double-booking guards ---")
    # Occupy a room, then try to double-book it for another admission
    rooms = None
    with db.get_connection() as conn:
        rooms = conn.execute("SELECT room_id FROM Operating_Rooms LIMIT 1").fetchone()
    room_id = rooms["room_id"]
    db.update_room_status(room_id, "Occupied")
    ok &= check(
        f"admission double-booking room #{room_id}",
        lambda: db.add_admission({"patient_id": 1, "doctor_id": 1, "room_id": room_id}),
    )
    db.update_room_status(room_id, "Available")  # cleanup

    with db.get_connection() as conn:
        bed = conn.execute("SELECT bed_id FROM ICU_Beds WHERE status='Available' LIMIT 1").fetchone()
        if not bed:
            # every bed happens to be occupied from a previous demo run -- release one to test with
            any_bed = conn.execute("SELECT bed_id FROM ICU_Beds LIMIT 1").fetchone()
            db.update_icu_bed(any_bed["bed_id"], patient_id=None)
            bed = any_bed
    bed_id = bed["bed_id"]
    db.update_icu_bed(bed_id, patient_id=1)  # occupy it for patient 1
    ok &= check(
        f"assigning already-occupied bed #{bed_id} to a different patient",
        lambda: db.update_icu_bed(bed_id, patient_id=2),
    )
    db.update_icu_bed(bed_id, patient_id=None)  # release / cleanup

    print("\n--- Hospitals.available_icu_beds self-healing ---")
    with db.get_connection() as conn:
        conn.execute("UPDATE Hospitals SET available_icu_beds = -999 WHERE hospital_id = 1")
        conn.commit()
    info = db.get_hospital_info(1)
    true_count_row = None
    with db.get_connection() as conn:
        true_count_row = conn.execute(
            "SELECT COUNT(*) AS n FROM ICU_Beds WHERE status='Available'"
        ).fetchone()
    if info["available_icu_beds"] == true_count_row["n"] and info["available_icu_beds"] != -999:
        print(f"[PASS] stale counter (-999) self-healed to real count ({info['available_icu_beds']})")
    else:
        print(f"[FAIL] counter did not self-heal: {info['available_icu_beds']}")
        ok = False

    print("\n--- Age bound consistency (0-120 everywhere) ---")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mcp_server"))
    import schemas
    import validation as val
    assert schemas.REGISTER_PATIENT_SCHEMA["properties"]["age"]["maximum"] == 120
    try:
        val.validate_patient_registration({"name": "Test", "age": 110, "gender": "Male"})
        print("[PASS] age=110 (previously rejected by the 0-100 mismatch) now validates correctly")
    except Exception as e:
        print(f"[FAIL] age=110 unexpectedly rejected: {e}")
        ok = False

    print("\n" + ("ALL DB FIXES VERIFIED" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
