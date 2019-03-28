 LOAD DATA LOCAL INFILE '~/Documents/googleplaystore_user_reviews.csv'  INTO TABLE reviews  FIELDS TERMINATED BY ','  ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 ROWS;
