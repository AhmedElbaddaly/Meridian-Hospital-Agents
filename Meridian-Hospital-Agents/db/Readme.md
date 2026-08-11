# Database Design - Meridian Hospital

This folder contains the database design and sample data used for the Meridian Hospital MCP Server project.

## Folder Contents

- `schema.sql`
  - Creates the database structure.
  - Defines tables, primary keys, foreign keys, and constraints.

- `seed.sql`
  - Inserts sample records used for testing and demonstrations.

- `erd.png`
  - Entity Relationship Diagram (ERD) that shows the database entities and their relationships.

## Database Overview

The database stores the main hospital management data required by the MCP server, including:

- Users (Admin, Doctors, Nurses)
- Patients and their medical information
- Hospitals and available resources
- ICU beds
- Operating rooms
- Admissions records

## Database Setup

1. Run `schema.sql` to create the database and tables.

2. Run `seed.sql` to insert sample data.

3. Use `erd.png` as a reference for understanding the database structure.

## Design Notes

- Primary keys uniquely identify each entity.
- Foreign keys maintain relationships between patients, doctors, admissions, rooms, and ICU beds.
- Constraints ensure valid values for roles, statuses, and resource availability.

## Security Considerations

The database is not accessed directly by the LLM.

All operations are handled through the MCP server, where:
- Read operations are exposed through controlled tools.
- Write operations require validation and authorization.
- Sensitive state changes are protected by server-side checks.
