#!/usr/bin/env python3
"""
agent/test_e2e.py
------------------
Fixed end-to-end tests covering protocol concerns from the agent's side.
"""
import asyncio
import sys

from agent import MediCoreAgent

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = ""):
    results.append((name, PASS if condition else FAIL, detail))
    print(f"[{PASS if condition else FAIL}] {name} {('- ' + detail) if detail and not condition else ''}")


async def test_capability_negotiation():
    agent = MediCoreAgent(auto_confirm=True)
    await agent.start()
    check("1. capability_negotiation: server declares elicitation+sampling+notifications",
          agent.supports("elicitation") and agent.supports("sampling")
          and agent.server_capabilities.get("tools", {}).get("listChanged") is True)
    await agent.stop()


async def test_notifications_tools_list_changed():
    agent = MediCoreAgent(auto_confirm=True)
    await agent.start()
    before = {t["name"] for t in agent.tools}
    check("2. notifications: write tools hidden before doctor login",
          "reserve_icu_bed" not in before and "create_admission" not in before)
    await agent.call_tool("login_as_doctor", {"doctor_id": "D001", "pin": "1234"})
    await asyncio.sleep(0.1)
    after = {t["name"] for t in agent.tools}
    check("2. notifications: tools/list_changed unlocked write tools after login",
          "reserve_icu_bed" in after and "create_admission" in after)
    await agent.stop()


async def test_elicitation_scarce_bed():
    agent = MediCoreAgent(auto_confirm=False)
    agent.scripted_answers = [True]
    await agent.start()
    await agent.call_tool("login_as_doctor", {"doctor_id": "D001", "pin": "1234"})
    await asyncio.sleep(0.1)
    result = await agent.call_tool("reserve_icu_bed", {"patient_id": "P002", "bed_id": "ICU-A1"})
    check("3. elicitation: last-bed reservation paused for human, then approved",
          not result.get("isError") and len(agent.scripted_answers) == 0,
          str(result))
    await agent.stop()


async def test_elicitation_declined_blocks_action():
    agent = MediCoreAgent(auto_confirm=False)
    agent.scripted_answers = [False]
    await agent.start()
    await agent.call_tool("login_as_doctor", {"doctor_id": "D001", "pin": "1234"})
    await asyncio.sleep(0.1)
    result = await agent.call_tool("update_patient_status", {"patient_id": "P002", "status": "deceased"})
    check("3b. elicitation: irreversible status change is BLOCKED when human declines",
          result.get("isError") is True, str(result))
    await agent.stop()


async def test_resources():
    agent = MediCoreAgent(auto_confirm=True)
    await agent.start()
    res = await agent.read_resource("policy://icu-admission")
    text = res.get("contents", [{}])[0].get("text", "")
    check("4. resources: ICU policy document readable via resources/read (not a tool)",
          "ICU Admission" in text)
    await agent.stop()


async def test_prompts():
    agent = MediCoreAgent(auto_confirm=True)
    await agent.start()
    prompt = await agent.get_prompt("triage_summary_for_admission", {"patient_id": "P001"})
    text = prompt.get("messages", [{}])[0].get("content", {}).get("text", "")
    check("5. prompts: parameterized triage_summary_for_admission template resolves patient_id",
          "P001" in text)
    await agent.stop()


async def test_progress_tracking():
    agent = MediCoreAgent(auto_confirm=True)
    progress_events = []
    original_handler = agent._handle_server_notification

    async def spy(method, params):
        if method == "notifications/progress":
            progress_events.append(params)
        await original_handler(method, params)

    agent._handle_server_notification = spy
    await agent.start()
    agent.endpoint.notification_handler = agent._handle_server_notification
    await agent.call_tool("list_hospitals_with_available_icu", {})
    check("6. progress_tracking: long-running scan reported >1 progress notification",
          len(progress_events) >= 2, f"got {len(progress_events)} events")
    await agent.stop()


async def test_defensive_tool_design():
    agent = MediCoreAgent(auto_confirm=True)
    await agent.start()
    unauth = await agent.call_tool("reserve_icu_bed", {"patient_id": "P001", "bed_id": "ICU-A1"})
    check("7a. defensive_design: write tool rejected without doctor authentication",
          unauth.get("isError") is True, str(unauth))

    await agent.call_tool("login_as_doctor", {"doctor_id": "D001", "pin": "1234"})
    await asyncio.sleep(0.1)
    already_taken = await agent.call_tool("reserve_icu_bed", {"patient_id": "P001", "bed_id": "ICU-A2"})
    check("7b. defensive_design: server rejects reserving an already-unavailable bed",
          already_taken.get("isError") is True, str(already_taken))
    await agent.stop()


async def test_sampling():
    agent = MediCoreAgent(auto_confirm=True)
    await agent.start()
    await agent.call_tool("login_as_doctor", {"doctor_id": "D001", "pin": "1234"})
    await asyncio.sleep(0.1)
    result = await agent.call_tool("create_admission",
                                    {"patient_id": "P002", "doctor_id": "D001", "room_id": "OR-1"})
    text = str(result)
    check("8. sampling: create_admission used the CLIENT's model to draft a justification",
          "Justification:" in text and "no sampling capability" not in text, text)
    await agent.stop()


async def main():
    await test_capability_negotiation()
    await test_notifications_tools_list_changed()
    await test_elicitation_scarce_bed()
    await test_elicitation_declined_blocks_action()
    await test_resources()
    await test_prompts()
    await test_progress_tracking()
    await test_defensive_tool_design()
    await test_sampling()

    print("\n--- summary ---")
    failed = [r for r in results if r[1] == FAIL]
    for name, status, detail in results:
        print(f"{status}: {name}")
    print(f"\n{len(results) - len(failed)}/{len(results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
