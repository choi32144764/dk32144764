INSERT INTO lecture(lecture_id, lecture_name, lecture_room)
VALUES('0', '캡스톤디자인1', '소융대');

INSERT INTO student(student_id, student_name)
VALUES('2014104149', '정해갑');
INSERT INTO student(student_id, student_name)
VALUES('2014104161', '허진호');
INSERT INTO student(student_id, student_name)
VALUES('2014104117', '양원영');
INSERT INTO student(student_id, student_name)
VALUES('2016100789', '이준오');

INSERT INTO lecture_students(lecture_id, student_id)
VALUES('0', '2014104149');
INSERT INTO lecture_students(lecture_id, student_id)
VALUES('0', '2014104161');
INSERT INTO lecture_students(lecture_id, student_id)
VALUES('0', '2014104117');
INSERT INTO lecture_students(lecture_id, student_id)
VALUES('0', '2016100789');