-- Top 100 documents by length
SELECT d.id,
  d.contents,
  d.LENGTH(contents) AS len_c
FROM "document" d
ORDER BY 3 DESC
LIMIT 100;
