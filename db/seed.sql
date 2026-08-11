USE meridian_hospital;

INSERT INTO Users (name, email, password_hash, role)
VALUES
('Ahmed Hassan','admin1@meridian.com','hash1','Admin'),
('Mona Ibrahim','admin2@meridian.com','hash2','Admin'),
('Dr. Sarah Ali','sarah@meridian.com','hash3','Doctor'),
('Dr. Omar Khaled','omar@meridian.com','hash4','Doctor'),
('Dr. Mariam Adel','mariam@meridian.com','hash5','Doctor'),
('Dr. Mostafa Nabil','mostafa@meridian.com','hash6','Doctor'),
('Nour Mohamed','nour@meridian.com','hash7','Nurse'),
('Salma Ali','salma@meridian.com','hash8','Nurse'),
('Youssef Tarek','youssef@meridian.com','hash9','Nurse'),
('Heba Samir','heba@meridian.com','hash10','Nurse');


INSERT INTO Patients 
(name, age, gender, blood_type, diagnosis, status)
VALUES
('Mohamed Adel',65,'Male','A+','Heart Attack','Waiting'),
('Sara Ahmed',58,'Female','O+','Stroke','ICU'),
('Ali Mahmoud',24,'Male','B+','Appendicitis','Surgery'),
('Fatma Hassan',40,'Female','AB+','Fracture','Admitted'),
('Youssef Ibrahim',70,'Male','O-','COVID-19','Waiting'),
('Mariam Samy',31,'Female','A-','Pneumonia','ICU'),
('Karim Nasser',19,'Male','B-','Trauma','Waiting'),
('Hoda Adel',55,'Female','O+','Kidney Failure','ICU'),
('Ahmed Salah',46,'Male','A+','Internal Bleeding','Surgery'),
('Laila Mohamed',36,'Female','AB-','Sepsis','Waiting'),
('Omar Hassan',50,'Male','A+','Diabetes Complication','Admitted'),
('Nour Ali',28,'Female','O-','Asthma','Discharged'),
('Khaled Samir',77,'Male','B+','Heart Failure','ICU'),
('Aya Mostafa',22,'Female','A-','Infection','Waiting'),
('Hassan Tarek',60,'Male','O+','Lung Disease','Admitted'),
('Reem Adel',35,'Female','AB+','Burn Injury','Surgery'),
('Mahmoud Ali',44,'Male','B+','Accident Trauma','Waiting'),
('Salma Hassan',68,'Female','O-','Stroke','ICU'),
('Tamer Said',30,'Male',NULL,NULL,'Waiting'),
('Nada Ibrahim',26,'Female','A+','Appendicitis','Admitted'),
('Mostafa Fathy',80,'Male','O+','Critical Heart Condition','ICU'),
('Dina Khaled',45,'Female','B-','Kidney Failure','Discharged'),
('Amr Nabil',33,'Male','AB+','Fracture','Admitted'),
('Heba Ali',52,'Female','A+','Severe Infection','Waiting'),
('Ziad Mohamed',15,'Male','O+','Emergency Trauma','Waiting');


INSERT INTO Operating_Rooms 
(room_number,status)
VALUES
('OR-01','Occupied'),
('OR-02','Available'),
('OR-03','Maintenance'),
('OR-04','Available'),
('OR-05','Occupied');


INSERT INTO Hospitals
(hospital_name,city,available_icu_beds)
VALUES
('Alex Care Hospital','Alexandria',5),
('El Salam Hospital','Alexandria',0),
('International Medical Center','Cairo',12),
('Future Hospital','Alexandria',2),
('Royal Hospital','Giza',8),
('El Hayat Hospital','Alexandria',0);


INSERT INTO ICU_Beds
(bed_number,status,patient_id)
VALUES
('ICU-01','Occupied',2),
('ICU-02','Occupied',6),
('ICU-03','Occupied',8),
('ICU-04','Occupied',13),
('ICU-05','Maintenance',NULL),
('ICU-06','Occupied',18),
('ICU-07','Occupied',21),
('ICU-08','Available',NULL),
('ICU-09','Available',NULL),
('ICU-10','Occupied',15);


INSERT INTO Admissions
(patient_id,doctor_id,room_id,admission_date,status)
VALUES
(1,3,NULL,GETDATE(),'Active'),
(2,4,NULL,GETDATE(),'Active'),
(3,3,1,GETDATE(),'Active'),
(4,5,NULL,GETDATE(),'Completed'),
(6,4,NULL,GETDATE(),'Active'),
(8,5,NULL,GETDATE(),'Active'),
(9,3,5,GETDATE(),'Active'),
(10,6,NULL,GETDATE(),'Transferred'),
(11,5,NULL,GETDATE(),'Active'),
(12,6,NULL,GETDATE(),'Completed'),
(13,4,NULL,GETDATE(),'Active'),
(15,3,NULL,GETDATE(),'Active'),
(16,6,1,GETDATE(),'Active'),
(18,5,NULL,GETDATE(),'Active'),
(20,3,NULL,GETDATE(),'Completed'),
(22,5,NULL,GETDATE(),'Completed'),
(21,4,NULL,GETDATE(),'Active'),
(19,3,NULL,GETDATE(),'Active'),
(23,6,NULL,GETDATE(),'Active'),
(25,4,NULL,GETDATE(),'Active');