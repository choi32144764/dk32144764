CREATE TABLE student(
student_id VARCHAR(10) NOT NULL,
student_name VARCHAR(50) NOT NULL,
PRIMARY KEY(student_id)
);
CREATE TABLE student_embedding(
student_id VARCHAR(10) NOT NULL,
embedding_date DATE NOT NULL,
embedding BLOB NOT NULL,
FOREIGN KEY (student_id) REFERENCES student(student_id)
);