#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "agent"))
sys.path.insert(0, os.path.join(ROOT, "db"))

from mcp_protocol import JsonRpcEndpoint, stdio_streams_for_this_process  # noqa: E402
import db_helpers as db  # noqa: E402


ICU_POLICY_DOC = (
    "Meridian ICU Admission & Prioritization Policy (v3):\n"
    "1. ICU beds are allocated by clinical acuity, not arrival order.\n"
    "2. When only one ICU bed remains network-wide, reservation requires\n"
    "   explicit attending physician sign-off (elicitation).\n"
    "3. Reassigning an Occupied operating room requires confirmation.\n"
    "4. Status changes to Deceased / Discharged_AMA require confirmation.\n"
)

TRIAGE_PROMPT_TEMPLATE = (
    "Summarize the clinical picture for patient {patient_id} in 2-3 sentences "
    "for the admitting physician, and state whether ICU-level care appears warranted."
)


def _ok(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict:
    return {"isError": True, "content": [{"type": "text", "text": text}]}


class DbBridgeServer:
    """Agent-protocol MCP server backed by Meridian SQLite (seed data)."""

    def __init__(self):
        self.endpoint = None
        self.role = "front_desk"
        self.client_capabilities = {}
        self.server_capabilities = {
            "tools": {"listChanged": True},
            "resources": {"listChanged": False},
            "prompts": {"listChanged": False},
            "elicitation": {},
            "sampling": {},
            "progress": {},
        }

    # ---- tool catalogs (same names the agent already knows) ----
    def read_tools(self):
        return [
            {"name": "get_patient", "description": "Retrieve patient from Meridian DB.",
             "inputSchema": {"type": "object",
                              "properties": {"patient_id": {"type": "string"}},
                              "required": ["patient_id"], "additionalProperties": False}},
            {"name": "list_available_icu_beds", "description": "List Available ICU beds from DB.",
             "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
            {"name": "list_available_operating_rooms", "description": "List Available ORs from DB.",
             "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
            {"name": "list_hospitals_with_available_icu",
             "description": "Scan hospitals with ICU capacity (progress notifications).",
             "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
        ]

    def write_tools(self):
        return [
            {"name": "reserve_icu_bed", "description": "Reserve ICU bed in DB for a patient.",
             "inputSchema": {"type": "object",
                              "properties": {"patient_id": {"type": "string"},
                                              "bed_id": {"type": "string"}},
                              "required": ["patient_id", "bed_id"], "additionalProperties": False}},
            {"name": "reserve_operating_room", "description": "Reserve OR in DB for a patient.",
             "inputSchema": {"type": "object",
                              "properties": {"patient_id": {"type": "string"},
                                              "room_id": {"type": "string"}},
                              "required": ["patient_id", "room_id"], "additionalProperties": False}},
            {"name": "create_admission", "description": "Insert admission row in DB.",
             "inputSchema": {"type": "object",
                              "properties": {"patient_id": {"type": "string"},
                                              "doctor_id": {"type": "string"},
                                              "room_id": {"type": "string"}},
                              "required": ["patient_id", "doctor_id", "room_id"],
                              "additionalProperties": False}},
            {"name": "update_patient_status", "description": "Update patient status in DB.",
             "inputSchema": {"type": "object",
                              "properties": {
                                  "patient_id": {"type": "string"},
                                  "status": {"type": "string"},
                              },
                              "required": ["patient_id", "status"], "additionalProperties": False}},
            {"name": "login_as_doctor",
             "description": "Auth as doctor (unlocks write tools, tools/list_changed).",
             "inputSchema": {"type": "object",
                              "properties": {"doctor_id": {"type": "string"}, "pin": {"type": "string"}},
                              "required": ["doctor_id", "pin"], "additionalProperties": False}},
        ]

    def visible_tools(self):
        return self.read_tools() + (self.write_tools() if self.role == "doctor" else [])

    async def handle_request(self, method: str, params: dict):
        if method == "initialize":
            self.client_capabilities = params.get("capabilities", {})
            return {
                "protocolVersion": "2025-06-18",
                "capabilities": self.server_capabilities,
                "serverInfo": {"name": "meridian-db-bridge", "version": "1.0.0"},
            }
        if method == "tools/list":
            return {"tools": self.visible_tools()}
        if method == "resources/list":
            return {"resources": [{"uri": "policy://icu-admission", "name": "ICU Admission Policy",
                                    "mimeType": "text/plain"}]}
        if method == "resources/read":
            if params.get("uri") == "policy://icu-admission":
                return {"contents": [{"uri": "policy://icu-admission", "mimeType": "text/plain",
                                       "text": ICU_POLICY_DOC}]}
            raise ValueError("unknown resource uri")
        if method == "prompts/list":
            return {"prompts": [{"name": "triage_summary_for_admission",
                                  "description": "Triage summary template",
                                  "arguments": [{"name": "patient_id", "required": True}]}]}
        if method == "prompts/get":
            pid = params.get("arguments", {}).get("patient_id", "{patient_id}")
            return {"messages": [{"role": "user", "content": {"type": "text",
                                   "text": TRIAGE_PROMPT_TEMPLATE.format(patient_id=pid)}}]}
        if method == "tools/call":
            return await self.call_tool(params.get("name"), params.get("arguments", {}))
        raise ValueError(f"unhandled method: {method}")

    async def handle_notification(self, method: str, params: dict):
        return

    async def call_tool(self, name: str, args: dict):
        write_names = {t["name"] for t in self.write_tools()}
        if name in write_names and name != "login_as_doctor" and self.role != "doctor":
            return _err("authorization denied: write tools require an authenticated doctor session")

        if name == "login_as_doctor":
            doctor = db.get_doctor_by_id_str(args.get("doctor_id", ""))
            if args.get("pin") == "1234" and doctor:
                self.role = "doctor"
                await self.endpoint.send_notification("notifications/tools/list_changed", {})
                return _ok(f"Authenticated as doctor {doctor['name']} (id={doctor['user_id']}). "
                           f"Write tools unlocked. [DB-backed]")
            return _err("invalid pin or doctor_id")

        if name == "get_patient":
            pid = db.parse_patient_id(args.get("patient_id"))
            p = db.get_patient(pid) if pid else None
            return _ok(str(p)) if p else _err("patient not found")

        if name == "list_available_icu_beds":
            return _ok(str(db.get_free_icu_beds()))

        if name == "list_available_operating_rooms":
            return _ok(str(db.get_available_operating_rooms()))

        if name == "list_hospitals_with_available_icu":
            hospitals = db.get_hospitals()
            found = []
            total = max(len(hospitals), 1)
            for i, h in enumerate(hospitals, start=1):
                await asyncio.sleep(0.12)
                await self.endpoint.send_notification(
                    "notifications/progress",
                    {"progress": i, "total": total,
                     "message": f"checked {h['hospital_name']} ({h['city']})"},
                )
                if (h.get("available_icu_beds") or 0) > 0:
                    found.append(h["hospital_name"])
            return _ok(str(found))

        if name == "reserve_icu_bed":
            pid = db.parse_patient_id(args.get("patient_id"))
            bed_key = str(args.get("bed_id", ""))
            bed = db.get_icu_bed_by_number(bed_key)
            if not bed and bed_key.isdigit():
                bed = db.get_icu_bed(int(bed_key))
            # tolerate agent demo IDs like ICU-B1 → map to first free bed
            if not bed and bed_key.upper().startswith("ICU-"):
                free = db.get_free_icu_beds()
                bed = free[0] if free else None
            if not bed:
                return _err("unknown bed_id")
            if bed["status"] != "Available":
                return _err("bed is not available")
            if not pid or not db.get_patient(pid):
                return _err("unknown patient_id")
            remaining = db.count_available_icu_at_hospital()
            if remaining <= 1:
                confirmed = await self.endpoint.send_request(
                    "elicitation/create",
                    {"message": f"This is the LAST available ICU bed ({bed['bed_number']}). "
                                f"Confirm reserving for patient {pid}?",
                     "requestedSchema": {"type": "object",
                                          "properties": {"confirm": {"type": "boolean"}},
                                          "required": ["confirm"]}},
                )
                if not confirmed.get("content", {}).get("confirm"):
                    return _err("reservation cancelled: human did not confirm")
            db.update_icu_bed(bed["bed_id"], pid)
            db.update_patient_status(pid, "ICU")
            return _ok(f"Reserved {bed['bed_number']} for patient {pid} [DB]")

        if name == "reserve_operating_room":
            pid = db.parse_patient_id(args.get("patient_id"))
            room_key = str(args.get("room_id", ""))
            room = db.get_operating_room_by_number(room_key)
            if not room and room_key.isdigit():
                room = db.get_operating_room(int(room_key))
            if not room:
                # map OR-1 style demo id → first available
                rooms = db.get_available_operating_rooms()
                room = rooms[0] if rooms else None
            if not room:
                return _err("unknown room_id")
            if room["status"] != "Available":
                confirmed = await self.endpoint.send_request(
                    "elicitation/create",
                    {"message": f"{room['room_number']} is {room['status']}. "
                                f"Confirm reassigning to patient {pid}?",
                     "requestedSchema": {"type": "object",
                                          "properties": {"confirm": {"type": "boolean"}},
                                          "required": ["confirm"]}},
                )
                if not confirmed.get("content", {}).get("confirm"):
                    return _err("reassignment cancelled: human did not confirm")
            db.reserve_operating_room(room["room_id"])
            return _ok(f"Reserved {room['room_number']} for patient {pid} [DB]")

        if name == "create_admission":
            pid = db.parse_patient_id(args.get("patient_id"))
            if not pid or not db.get_patient(pid):
                return _err("unknown patient_id")
            doctor = db.get_doctor_by_id_str(args.get("doctor_id", ""))
            if not doctor:
                return _err("unknown doctor_id")
            room = db.get_operating_room_by_number(str(args.get("room_id", "")))
            room_id = room["room_id"] if room else None
            justification = "(no sampling capability on this client)"
            if "sampling" in self.client_capabilities:
                patient = db.get_patient(pid)
                sample = await self.endpoint.send_request(
                    "sampling/createMessage",
                    {"messages": [{"role": "user", "content": {"type": "text",
                        "text": f"In one sentence, write a clinical admission justification for "
                                f"{patient['name']} ({patient.get('diagnosis') or 'n/a'})."}}],
                     "maxTokens": 100},
                )
                justification = sample.get("content", {}).get("text", justification)
            admission_id = db.add_admission({
                "patient_id": pid,
                "doctor_id": doctor["user_id"],
                "room_id": room_id,
                "status": "Active",
            })
            db.update_patient_status(pid, "Admitted")
            return _ok(f"Created admission {admission_id}. Justification: {justification}")

        if name == "update_patient_status":
            pid = db.parse_patient_id(args.get("patient_id"))
            status = args.get("status", "")
            # map agent demo enums → DB enums
            status_map = {
                "waiting_admission": "Waiting",
                "admitted": "Admitted",
                "in_surgery": "Surgery",
                "discharged": "Discharged",
                "deceased": "Deceased",
                "discharged_against_medical_advice": "Discharged_AMA",
            }
            status = status_map.get(status, status)
            if not pid or not db.get_patient(pid):
                return _err("unknown patient_id")
            if status in ("Deceased", "Discharged_AMA"):
                confirmed = await self.endpoint.send_request(
                    "elicitation/create",
                    {"message": f"Setting status to '{status}' is irreversible. Confirm for patient {pid}?",
                     "requestedSchema": {"type": "object",
                                          "properties": {"confirm": {"type": "boolean"}},
                                          "required": ["confirm"]}},
                )
                if not confirmed.get("content", {}).get("confirm"):
                    return _err("status change cancelled: human did not confirm")
            db.update_patient_status(pid, status)
            return _ok(f"Updated patient {pid} status to {status} [DB]")

        return _err(f"unknown tool: {name}")


async def main():
    reader, writer = await stdio_streams_for_this_process()
    server = DbBridgeServer()
    endpoint = JsonRpcEndpoint(
        reader, writer,
        request_handler=server.handle_request,
        notification_handler=server.handle_notification,
        name="db-bridge",
    )
    server.endpoint = endpoint
    await endpoint.run()


if __name__ == "__main__":
    asyncio.run(main())
