-- Top 100 documents by length
SELECT d.id,
  d.contents,
  d.LENGTH(contents) AS len_c
FROM "document" d
ORDER BY 3 DESC
LIMIT 100;

-- Top 100 documents by length
SELECT d.id,
  d.contents,
  LENGTH(d.contents) AS len_c
FROM public.document d
WHERE d.contents NOT LIKE '%Title: Pandemic (H1N1) 2009%'
  AND d.contents NOT LIKE '%Title: Severe Acute Respiratory Syndrome (SARS)%'
ORDER BY 3 DESC
LIMIT 100;

-- Relevant documents for BM25
-- Manually curated from the site search
select contents, url, published_at, meta
from v_doc_embedding
where contents like 'Title: Ebola%'
and published_at >= '2021-01-01'
and right(meta->>'don_id', 6) in ('DON433','DON428','DON425','DON423','DON421','DON411','DON410','DON404','DON398','DON377','DON351','DON328','DON325','DON312','DON310','ongo_1')
and batch = 1
and model = 'all-minilm';