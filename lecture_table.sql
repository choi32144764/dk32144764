CREATE TABLE lecture(
lecture_id VARCHAR(20) NOT NULL,
lecture_name VARCHAR(50),
PRIMARY KEY(lecture_id)
);
CREATE TABLE lecture_students(
lecture_id VARCHAR(20) NOT NULL,
student_id VARCHAR(10) UNIQUE,
FOREIGN KEY (lecture_id) REFERENCES lecture(lecture_id),
FOREIGN KEY (student_id) REFERENCES student(student_id)
);
CREATE TABLE student_attendance(
lecture_id VARCHAR(20) NOT NULL,
student_id VARCHAR(10) NOT NULL,
status VARCHAR(10) NOT NULL,
attendance_time DATETIME DEFAULT CURRENT_TIMESTAMP, 
FOREIGN KEY (lecture_id) REFERENCES lecture(lecture_id),
FOREIGN KEY (student_id) REFERENCES student(student_id)
);
CREATE TABLE undefined_image(
lecture_id VARCHAR(20) NOT NULL,
image MEDIUMBLOB NOT NULL,
width SMALLINT NOT NULL,
height SMALLINT NOT NULL,
timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
FOREIGN KEY (lecture_id) REFERENCES lecture(lecture_id)
);
CREATE TABLE lecture_schedule(
lecture_id VARCHAR(20) NOT NULL,
lecture_day TINYINT NOT NULL,
lecture_room VARCHAR(50) NOT NULL,
lecture_start_time TIME NOT NULL,
lecture_end_time TIME NOT NULL,
FOREIGN KEY (lecture_id) REFERENCES lecture(lecture_id)
);