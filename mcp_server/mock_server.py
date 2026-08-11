#!/usr/bin/env python3
"""
mcp_server/mock_server.py

Mock MCP Server used for end-to-end testing.
It simulates MCP.py behavior without database dependency.
"""
import asyncio
import os
import sys


if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )


sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "agent")
)

from mcp_protocol import (              # noqa
    JsonRpcEndpoint,
    stdio_streams_for_this_process
)


PATIENTS = {
    1: {
        "patient_id": 1,
        "name": "John Doe",
        "age": 45,
        "status": "Waiting"
    }
}


ICU_BEDS = {
    1: {
        "bed_id": 1,
        "status": "Available"
    },
    2: {
        "bed_id": 2,
        "status": "Available"
    }
}


ADMISSIONS = []


class MeridianMockServer:

    def __init__(self):

        self.endpoint = None

        self.client_capabilities = {}

        self.server_capabilities = {
            "tools": {
                "listChanged": True
            },
            "resources": {},
            "prompts": {},
            "elicitation": {},
            "sampling": {},
            "progress": {}
        }

    # -----------------------------
    # MCP TOOL DEFINITIONS
    # -----------------------------

    def tools(self):

        return [

            {
                "name": "register_patient",
                "description":
                "Register a new patient.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string"
                        },
                        "age": {
                            "type": "integer"
                        },
                        "gender": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "name",
                        "age",
                        "gender"
                    ],
                    "additionalProperties": False
                }
            },


            {
                "name": "update_patient_status",
                "description":
                "Update patient medical status.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "integer"
                        },
                        "status": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "patient_id",
                        "status"
                    ],
                    "additionalProperties": False
                }
            },


            {
                "name": "get_patient_details",
                "description":
                "Retrieve patient details.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "integer"
                        }
                    },
                    "required": [
                        "patient_id"
                    ],
                    "additionalProperties": False
                }
            },


            {
                "name": "create_admission",
                "description":
                "Create patient admission.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {
                            "type": "integer"
                        },
                        "doctor_id": {
                            "type": "integer"
                        }
                    },
                    "required": [
                        "patient_id",
                        "doctor_id"
                    ],
                    "additionalProperties": False
                }
            },


            {
                "name": "manage_icu_bed",
                "description":
                "Assign or release ICU bed.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "bed_id": {
                            "type": "integer"
                        },
                        "patient_id": {
                            "type": [
                                "integer",
                                "null"
                            ]
                        }
                    },
                    "required": [
                        "bed_id"
                    ],
                    "additionalProperties": False
                }
            },


            {
                "name": "get_available_icu_beds",
                "description":
                "Get available ICU beds.",
                "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            },


            {
                "name": "get_hospital_capacity",
                "description":
                "Check hospital capacity.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "hospital_id": {
                            "type": "integer"
                        }
                    },
                    "additionalProperties": False
                }
            }

        ]

    # -----------------------------
    # REQUEST HANDLER
    # -----------------------------

    async def handle_request(
            self,
            method,
            params):

        if method == "initialize":

            self.client_capabilities = (
                params.get(
                    "capabilities",
                    {}
                )
            )

            return {
                "protocolVersion":
                "2025-06-18",

                "capabilities":
                self.server_capabilities,

                "serverInfo": {
                    "name":
                    "meridian-mock-server",

                    "version":
                    "1.0"
                }
            }

        if method == "tools/list":

            return {
                "tools":
                self.tools()
            }

        if method == "resources/list":

            return {
                "resources": [
                    {
                        "uri":
                        "triage://protocols/guidelines",

                        "name":
                        "Triage Guidelines",

                        "mimeType":
                        "text/plain"
                    }
                ]
            }

        if method == "resources/read":

            return {
                "contents": [
                    {
                        "uri":
                        params["uri"],

                        "mimeType":
                        "text/plain",

                        "text":
                        "Emergency triage guidelines mock resource."
                    }
                ]
            }

        if method == "prompts/list":

            return {
                "prompts": [
                    {
                        "name":
                        "triage_patient_prompt",

                        "description":
                        "Triage assistant prompt"
                    }
                ]
            }

        if method == "prompts/get":

            return {
                "messages": [
                    {
                        "role":
                        "user",

                        "content": {
                            "type":
                            "text",

                            "text":
                            "Evaluate patient urgency and use hospital tools."
                        }
                    }
                ]
            }

        if method == "tools/call":

            return await self.call_tool(
                params["name"],
                params.get(
                    "arguments",
                    {}
                )
            )

        raise ValueError(
            f"Unknown method {method}"
        )

    # -----------------------------
    # TOOL EXECUTION MOCK
    # -----------------------------

    async def call_tool(
            self,
            name,
            args):

        if name == "register_patient":

            return {
                "content": [
                    {
                        "type":
                        "text",

                        "text":
                        "Mock: Patient registered successfully with ID 101"
                    }
                ]
            }

        if name == "get_patient_details":

            patient = PATIENTS.get(
                args["patient_id"]
            )

            return {
                "content": [
                    {
                        "type":
                        "text",

                        "text":
                        str(patient)
                    }
                ]
            }

        if name == "get_available_icu_beds":

            await self.endpoint.send_notification(
                "notifications/progress",
                {
                    "progress": 1,
                    "total": 1,
                    "message":
                    "Checking ICU beds"
                }
            )

            return {
                "content": [
                    {
                        "type":
                        "text",

                        "text":
                        str(ICU_BEDS)
                    }
                ]
            }

        if name == "manage_icu_bed":

            await self.endpoint.send_request(
                "elicitation/create",
                {
                    "message":
                    "Confirm ICU bed assignment?",
                    "requestedSchema": {
                        "type":
                        "object",
                        "properties": {
                            "confirm": {
                                "type":
                                "boolean"
                            }
                        }
                    }
                }
            )

            return {
                "content": [
                    {
                        "type":
                        "text",

                        "text":
                        "Mock: ICU bed assigned."
                    }
                ]
            }

        if name == "create_admission":

            if "sampling" in self.client_capabilities:

                await self.endpoint.send_request(
                    "sampling/createMessage",
                    {
                        "messages": [
                            {
                                "role":
                                "user",

                                "content": {
                                    "type":
                                    "text",

                                    "text":
                                    "Write admission justification."
                                }
                            }
                        ]
                    }
                )

            ADMISSIONS.append(args)

            return {
                "content": [
                    {
                        "type":
                        "text",

                        "text":
                        "Mock: Admission created successfully."
                    }
                ]
            }

        if name == "update_patient_status":

            return {
                "content": [
                    {
                        "type":
                        "text",

                        "text":
                        "Mock: Patient status updated."
                    }
                ]
            }

        if name == "get_hospital_capacity":

            return {
                "content": [
                    {
                        "type":
                        "text",

                        "text":
                        "Mock: Meridian Hospital ICU capacity = 8"
                    }
                ]
            }

        return {
            "isError":
            True,

            "content": [
                {
                    "type":
                    "text",

                    "text":
                    f"Unknown tool {name}"
                }
            ]
        }

    async def handle_notification(
            self,
            method,
            params):

        pass


async def main():

    reader, writer = await stdio_streams_for_this_process()

    server = MeridianMockServer()

    endpoint = JsonRpcEndpoint(
        reader,
        writer,
        request_handler=server.handle_request,

        notification_handler=server.handle_notification,

        name="server"
    )

    server.endpoint = endpoint

    await endpoint.run()


if __name__ == "__main__":

    asyncio.run(main())
