CREATE DATABASE meridian_hospital;
USE meridian_hospital;

CREATE TABLE Users (
    user_id INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL
        CHECK (role IN ('Admin','Doctor','Nurse'))
);

CREATE TABLE Patients (
    patient_id INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    gender VARCHAR(10)
        CHECK (gender IN ('Male','Female')),
    blood_type VARCHAR(5),
    diagnosis VARCHAR(255),
    status VARCHAR(20)
        DEFAULT 'Waiting'
        CHECK (status IN ('Waiting','Admitted','ICU','Surgery','Discharged'))
);

CREATE TABLE Operating_Rooms (
    room_id INT IDENTITY(1,1) PRIMARY KEY,
    room_number VARCHAR(10) UNIQUE NOT NULL,
    status VARCHAR(20)
        DEFAULT 'Available'
        CHECK (status IN ('Available','Occupied','Maintenance'))
);

CREATE TABLE Hospitals (
    hospital_id INT IDENTITY(1,1) PRIMARY KEY,
    hospital_name VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    available_icu_beds INT DEFAULT 0
);

CREATE TABLE Admissions (
    admission_id INT IDENTITY(1,1) PRIMARY KEY,
    patient_id INT NOT NULL,
    doctor_id INT NOT NULL,
    room_id INT NULL,
    admission_date DATETIME DEFAULT GETDATE(),
    status VARCHAR(20)
        DEFAULT 'Active'
        CHECK (status IN ('Active','Completed','Transferred')),

    CONSTRAINT FK_Admission_Patient
        FOREIGN KEY (patient_id)
        REFERENCES Patients(patient_id),

    CONSTRAINT FK_Admission_Doctor
        FOREIGN KEY (doctor_id)
        REFERENCES Users(user_id),

    CONSTRAINT FK_Admission_Room
        FOREIGN KEY (room_id)
        REFERENCES Operating_Rooms(room_id)
);

CREATE TABLE ICU_Beds (
    bed_id INT IDENTITY(1,1) PRIMARY KEY,
    bed_number VARCHAR(10) UNIQUE NOT NULL,
    status VARCHAR(20)
        DEFAULT 'Available'
        CHECK (status IN ('Available','Occupied','Maintenance')),
    patient_id INT NULL,

    CONSTRAINT FK_Bed_Patient
        FOREIGN KEY (patient_id)
        REFERENCES Patients(patient_id)
);
