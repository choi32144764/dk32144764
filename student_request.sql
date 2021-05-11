CREATE TABLE `student_request` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `lecture_id` varchar(20) NOT NULL,
  `student_id` varchar(10) NOT NULL,
  `student_name` varchar(45) DEFAULT NULL,
  `timestamp` datetime DEFAULT CURRENT_TIMESTAMP,
  `status` varchar(10) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `lecture_id` (`lecture_id`),
  KEY `student_id` (`student_id`),
  CONSTRAINT `student_request_ibfk_3` FOREIGN KEY (`lecture_id`) REFERENCES `lecture` (`lecture_id`),
  CONSTRAINT `student_request_ibfk_4` FOREIGN KEY (`student_id`) REFERENCES `student` (`student_id`)
) ENGINE=InnoDB AUTO_INCREMENT=29 DEFAULT CHARSET=utf8;