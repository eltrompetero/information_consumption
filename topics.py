# ====================================================================================== #
# Module for manipulating, analyzing, and simulating with the topic graph.
# Author: Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
import networkx as nx
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from multiprocess import Pool

from .firehose import *

@lru_cache
def all_topics():
    """Unique topics in database as extracted. Counts are returned in
    count_topics().

    Returns
    -------
    int
    """
    q = f'''{DEFAULT_INIT_PARAMS}
        SELECT DISTINCT topic_1 as topic
        from parquet_scan({firehose_days(as_file=True)})
        '''
    conn = db_conn()
    df = conn.execute(q).fetchdf()
    return df['topic'].values

@lru_cache
def count_topics():
    """No. of unique topics in database.

    Returns
    -------
    int
    """
    q = f'''{DEFAULT_INIT_PARAMS}
         CREATE TABLE freq(
             topic varchar(255),
         );

         INSERT INTO freq
             SELECT DISTINCT(topic_1) AS topic
             FROM parquet_scan('{FIREHOSE_PATH}/201806*/Redacted_Firehose_article_id.parquet')
         UNION ALL
             SELECT DISTINCT(topic_2) AS topic
             FROM parquet_scan('{FIREHOSE_PATH}/201806*/Redacted_Firehose_article_id.parquet');

         SELECT COUNT(DISTINCT topic)
         FROM freq
         '''
    return db_conn().execute(q).fetchdf().values[0][0]