## Background

Reddit is an online discussion board with a majority of its users from Western countries (most of whom North Americans). Threads belong to various "subreddits" based on topics, and there are five most active subreddits focusing on the Greater China Region: [r/china](https://reddit.com/r/china), [r/beijing](https://www.reddit.com/r/beijing), [r/shanghai](https://www.reddit.com/r/shanghai), [r/HongKong](https://www.reddit.com/r/HongKong), and [r/taiwan](https://www.reddit.com/r/taiwan).

Authors of this project are international (Chinese) students in the US, interested in foreigners' opinion about China and the broader Greater China Region. They decide to use techniques from their Text Mining class, and explore topics in the aforementioned subreddits.

## Objective(s) and methodology

- Supervised learning - tagging posts with "link flairs"

Some posts have been tagged by their OP with "flairs", which is Reddit's way of saying tags. We use those

- Unsupervised learning - topic modeling with LDA

Latent Dirichlet Allocation (**LDA**)

## Technologies and reasons of choice

- Microsoft Azure
  - Data Factory, for ETL `ndjson` into SQL Database. This tool has an easy-to-use GUI to compose pipelines, much like SQL Server Integration Service (**SSIS**). It is *hard* to debug, however.
  - SQL Database. We choose to use traditional relational database management system (**RDBMS**) because we need to *join* posts with comments, as well as filtering on various columns. There is no obvious advantage of using a document database (e.g. MongoDB), only obvious *disadvantages*.
  
- Apache Spark (on Azure HDInsight). 
- NLTK (Natural Language ToolKit). 

## Data and pre-processing

Jason maintains [a regularly updated repository](https://files.pushshift.io/reddit/) of Reddit data, organized by time. However, since this analysis only requires a small number of subreddits, we choose to use [the pre-made BigQuery dataset](https://www.reddit.com/r/bigquery/comments/3cej2b/17_billion_reddit_comments_loaded_on_bigquery/), which makes filtering based on subreddit a breeze.

For the sake of this project, we exclude "link" posts and posts without any comments, based on two considerations:

1. We need to follow the link and scrape the website in order to know what the link is about. Scraping exposes us to great legal risk.
2. Posts without comments are often too short for analysis; we also deem those as "not interested by others", thus of little value for understanding the mainstream opinion.
 
We then concatenate comments with the main post to form a single "document" for each thread, using *SQL*:

```sql  
select id, subreddit, score, author, created_utc, body + cmt as body, author_flair_text, link_flair_text into documents 
from (
  select posts.*, string_agg(comments.body,'.') as cmt 
  from posts 
  join comments on posts.subreddit = comments.subreddit and posts.id = comments.parent
  group by posts.id, posts.subreddit, posts.score, posts.author, posts.created_utc, posts.body, posts.author_flair_text, posts.link_flair_text
) as p; -- String_agg is a function on SQL Server 2017. In previous versions you might need to use some function like xml_path.
```

> ### Why use SQL for concatenation?
> The authors do have a few options, including writing their own Python script (they are most familiar with Python). However, they believe using a native function in RDBMS has a performance advantage because DBMS is written in C++ and highly optimized (batch operation, parallelism, etc.).

## Contact

> Mention of a brand, product, or service does not suggest any endorsement by the author group.