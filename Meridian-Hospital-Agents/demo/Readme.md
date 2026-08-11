# Project Demo

This demo demonstrates the MCP Agent execution workflow.

## Demo Overview

The recording shows that the MCP Agent can:

- Connect to the MCP server.
- Discover available MCP tools.
- Select the appropriate tool based on the user request.
- Execute MCP tools successfully.
- Return structured results.

## Available MCP Tools

The MCP server provides the following tools:

- `register_patient`
- `update_patient_status`
- `get_patient_details`
- `create_admission`
- `update_operating_room_status`
- `manage_icu_bed`
- `get_available_icu_beds`
- `get_hospital_capacity`

## Demonstrated Tool Calls

The demo includes successful execution of the following requests:

### 1. Get Available ICU Beds

**User Request**
```text
Which ICU beds are available?
```

**Selected Tool**
```text
get_available_icu_beds
```

**Result**
```text
Available ICU beds returned successfully.
```

### 2. Get Patient Details

**User Request**
```text
Get patient details
```

**Selected Tool**
```text
get_patient_details
```

**Result**
```text
Patient details retrieved successfully.
```

### 3. Create Admission

**User Request**
```text
Create admission
```

**Selected Tool**
```text
create_admission
```

**Result**
```text
Admission created successfully.
```

### 4. Get Hospital Capacity

**User Request**
```text
Get hospital capacity
```

**Selected Tool**
```text
get_hospital_capacity
```

**Result**
```text
Hospital capacity information returned successfully.
```

## Execution Flow

```text
User Request
      ↓
MCP Agent
      ↓
Tool Discovery
      ↓
Tool Selection
      ↓
MCP Tool Execution
      ↓
Structured Response
```

## Demo Evidence

The terminal recording shows real execution traces including:

- User queries
- Available MCP tools discovered from the server
- Selected tool calls
- Tool inputs and outputs
- Successful responses

The execution logs confirm that the MCP Agent successfully communicates with the MCP server, discovers available tools, selects the correct tools, and executes backend operations successfully.

## Recording

The demo recording is included in this folder.
