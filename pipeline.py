# ====================================================================================== #
# Pipeline methods for final analysis.
# 
# Author : Eddie Lee, edlee@csh.ac.at
# ====================================================================================== #
from workspace.utils import save_pickle
from misc.stats import DiscretePowerLaw

from .firehose import *
from .topics import count_topics
from .analysis import InfoFillHypothesis


def setup(force=False, setup_rlvcy=True):
    """Run all methods that rely on firehose data caching.

    Parameters
    ----------
    force : bool, False
        If True, rewrite all cached files.
    setup_rlvcy : bool, True
        If True, rerun filtering of domains by average relevancy scores.
    """
    if setup_rlvcy:
        filter_relevancy()
        setup_article_id()

    for day in firehose_days():
        article_topic(day, force=force)
        print(f"Done with {day}.")

    firm_topics()
    firm_source()
    firm_article()
    read_and_econ_df()

    print("Done with reading diversity scaling.")

def capital_scaling():
    q = f'''{DEFAULT_INIT_PARAMS}
        CREATE TABLE compustat AS
        SELECT domain, asset, plantpropertyequipment, sales, annual_employees
        FROM parquet_scan('../data/compustat_quarterly_size.parquet') comp
        WHERE quarter = '2018-07-01';

        SELECT *
        FROM (SELECT records.domain,
                     COUNT(records.domain) AS records,
                     COUNT(DISTINCT records.sources) AS sources
              FROM parquet_scan('{FIREHOSE_PATH}/201806*/Redacted_Firehose_article_id.parquet') records
              INNER JOIN compustat
                  ON records.domain = compustat.domain
              GROUP BY records.domain) records
        INNER JOIN compustat
           ON compustat.domain = records.domain
        '''
    df = db_conn().execute(q).fetchdf()
    save_pickle(['df'], 'cache/capital_scaling.p', True)

def power_law_fit(day='201806*'):
    conn = db.connect()

    # fit articles
    q = f'''{DEFAULT_INIT_PARAMS}
        SELECT domain, CAST(counts AS integer) AS counts
        FROM (SELECT domain, SUM(counts) AS counts
              FROM parquet_scan('{FIREHOSE_PATH}/{day}/uarticle.pq') alldata
              GROUP BY domain)
        WHERE counts>1
        '''
    art_counts = conn.sql(q).fetchdf()

    dpl = DiscretePowerLaw(alpha=2.)
    y = art_counts['counts'].values
    alpha, lb = dpl.max_likelihood(y, lower_bound_range=(2,1000))
    f = (y>=lb).mean()
    dpl = DiscretePowerLaw(alpha=alpha, lower_bound=lb)
    ksval = dpl.ksval(y[y>=lb], alpha=alpha, lower_bound=lb)
    p, ks_samp, (alpha_samp, lb_samp) = dpl.clauset_test(y[y>=lb], ksval,
                                                         lower_bound_range=(2,1000),
                                                         samples_below_cutoff=y[y<lb],
                                                         return_all=True)
    article_pow_fit = {'alpha':alpha, 'lb':lb, 'p':p, 'ks_samp':ks_samp,
                       'alpha_samp':alpha_samp,
                       'lb_samp':lb_samp,
                       'f':f}

    # fit sources
    q = f'''
        SELECT domain, CAST(counts AS integer) AS counts
        FROM (SELECT domain, SUM(counts) AS counts
              FROM parquet_scan('{FIREHOSE_PATH}/{day}/usource.pq') alldata
              GROUP BY domain)
        WHERE counts>1
        '''
    source_counts = conn.sql(q).fetchdf()

    dpl = DiscretePowerLaw(alpha=2.)
    y = source_counts['counts'].values
    alpha, lb = dpl.max_likelihood(y, lower_bound_range=(2,1000))
    f = (y>=lb).mean()
    dpl = DiscretePowerLaw(alpha=alpha, lower_bound=lb)
    ksval = dpl.ksval(y[y>=lb], alpha=alpha, lower_bound=lb)
    p, ks_samp, (alpha_samp, lb_samp) = dpl.clauset_test(y[y>=lb], ksval,
                                  lower_bound_range=(2,1000),
                                  samples_below_cutoff=y[y<lb],
                                  return_all=True)
    source_pow_fit = {'alpha':alpha, 'lb':lb, 'p':p, 'ks_samp':ks_samp,
                      'alpha_samp':alpha_samp,
                      'lb_samp':lb_samp,
                      'f':f}

    # fit records
    rec_counts = firm_records()

    dpl = DiscretePowerLaw(alpha=2.)
    y = rec_counts['counts'].values
    alpha, lb = dpl.max_likelihood(y, lower_bound_range=(2,1000))
    f = (rec_counts['counts']>=lb).mean()
    dpl = DiscretePowerLaw(alpha=alpha, lower_bound=lb)
    ksval = dpl.ksval(y[y>=lb], alpha=alpha, lower_bound=lb)
    p, ks_samp, (alpha_samp, lb_samp) = dpl.clauset_test(y[y>=lb], ksval,
                                  lower_bound_range=(2,1000),
                                  samples_below_cutoff=y[y<lb],
                                  return_all=True)
    record_pow_fit = {'alpha':alpha, 'lb':lb, 'p':p, 'ks_samp':ks_samp,
                      'alpha_samp':alpha_samp,
                      'lb_samp':lb_samp,
                      'f':f}


    # fit topics
    q = f'''
        SELECT domain, COUNT(DISTINCT topic) AS counts
        FROM (SELECT *
              FROM parquet_scan('./cache/utopics.pq'))
        GROUP BY domain
        '''
    topic_counts = conn.sql(q).fetchdf()

    dpl = DiscretePowerLaw(lower_bound=10, upper_bound=count_topics())
    y = topic_counts['counts'][topic_counts['counts']>=10].values
    alpha = dpl.max_likelihood(y, lower_bound=10, upper_bound=count_topics())
    f = (topic_counts['counts']>=10).mean()
    dpl = DiscretePowerLaw(alpha=alpha, lower_bound=10, upper_bound=count_topics())
    ksval = dpl.ksval(y)
    p, ks_samp, (alpha_samp, lb_samp) = dpl.clauset_test(y, ksval,
                                                         return_all=True)
    topic_pow_fit = {'alpha':alpha, 'lb':10, 'p':p, 'ks_samp':ks_samp,
                     'alpha_samp':alpha_samp,
                     'lb_samp':lb_samp,
                     'f':f}

    save_pickle(['article_pow_fit','source_pow_fit','record_pow_fit','topic_pow_fit'],
                 './cache/pow_fits.p', True)

def us_gov_overlap_stats(day='201806*'):
    """Print some basic stats about how much of data pertains to US govt reading.
    """
    conn = db_conn()

    print(f"On days of {day}, we have")
    q = f'''{DEFAULT_INIT_PARAMS}
         SELECT COUNT(*)
         FROM parquet_scan('{FIREHOSE_PATH}/{day}/Redacted_Firehose_article_id.parquet')
         '''
    print(f"{conn.execute(q).fetchdf().values[0][0]} records")

    q = f'''
         SELECT COUNT(DISTINCT domain)
         FROM parquet_scan('{FIREHOSE_PATH}/{day}/Redacted_Firehose_article_id.parquet')
         '''
    print(f"{conn.execute(q).fetchdf().values[0][0]} distinct domains")

    gov_url_df = pd.read_csv('../data/us_gov_url/1_govt_urls_full.csv')
    print(f"{gov_url_df.shape[0]} gov URLs identified in repo")
    print()

    q = f'''
         SELECT COUNT(DISTINCT fire.domain)
         FROM parquet_scan('{FIREHOSE_PATH}/{day}/Redacted_Firehose_article_id.parquet') AS fire
         INNER JOIN gov_url_df AS gov
            ON gov.Domain = fire.domain
         '''
    print(f"{conn.execute(q).fetchdf().values[0][0]} domains in US gov repo appear in firehose")

    q = f'''
         SELECT COUNT(DISTINCT domain)
         FROM parquet_scan('{FIREHOSE_PATH}/{day}/Redacted_Firehose_article_id.parquet')
         WHERE domain LIKE '%.gov' OR domain LIKE '%.mil'
         '''
    print(f"{conn.execute(q).fetchdf().values[0][0]} domains with endings .gov or .mil appear in firehose")
    print()

    q = f'''
         SELECT COUNT(*)
         FROM parquet_scan('{FIREHOSE_PATH}/{day}/Redacted_Firehose_article_id.parquet') AS fire
            INNER JOIN (SELECT Domain
                        FROM gov_url_df) AS gov
            ON fire.domain=gov.Domain
         '''
    print(f"{conn.execute(q).fetchdf().values[0][0]} records are from US govt URLs")

    q = f'''
         SELECT COUNT(*)
         FROM parquet_scan('{FIREHOSE_PATH}/{day}/Redacted_Firehose_article_id.parquet')
         WHERE domain LIKE '%.gov' OR domain LIKE '%.mil'
         '''
    print(f"{conn.execute(q).fetchdf().values[0][0]} records are from .gov or .mil URLs")
    print()

    q = f'''
         SELECT COUNT(DISTINCT domain)
         FROM parquet_scan('{FIREHOSE_PATH}/{day}/Redacted_Firehose_article_id.parquet')
         WHERE domain LIKE '%.edu' OR domain LIKE '%.edu.%'
         '''
    print(f"{conn.execute(q).fetchdf().values[0][0]} domains with endings containing .edu")

    q = f'''
         SELECT COUNT(*)
         FROM parquet_scan('{FIREHOSE_PATH}/{day}/Redacted_Firehose_article_id.parquet')
         WHERE domain LIKE '%.edu' OR domain LIKE '%.edu.%'
         '''
    print(f"These constitute {conn.execute(q).fetchdf().values[0][0]} records")

def basic_info():
    """Some overall figures from the data including the total number of firms we have
    and a histogram of the number of firms that have read a certain number of topics.
    """
    nfirms = []  # no. of firms on each day
    ntopics = []  # no. of topics covered by each firm by day

    for day in range(10, 24):
        conn = db_conn()
        q = f'''{DEFAULT_INIT_PARAMS}
            CREATE TABLE firm_topics AS
            SELECT *
            FROM parquet_scan('{FIREHOSE_PATH}/201806{day}/utopics.pq');
            '''
        conn.execute(q)

        q = f'''SELECT COUNT(DISTINCT domain)
            FROM firm_topics
            '''
        nfirms.append(conn.execute(q).fetchdf().values[0][0])

        q = f'''SELECT domain, COUNT(*) AS counts
            FROM firm_topics
            GROUP BY domain
            '''
        ntopics.append(conn.execute(q).fetchdf()['counts'].values)
    save_pickle(['nfirms','ntopics'], 'cache/basic_info.p', True)

def devices2employees():
    """Number of devices with employee count."""
    q = f'''CONFIDENTIAL QUERY
        '''
    df = db_conn().execute(q).fetchdf()
    save_pickle(['df'], 'cache/devices2employees.p', True)

def scaling_deviations():
    """Output parquet files for regression analyses."""
    with open('cache/pow_fits.p', 'rb') as f:
        article_pow_fit, source_pow_fit, record_pow_fit, topic_pow_fit = list(pickle.load(f).values()) 
    econdf = read_and_econ_df()

    for info_type in ['records', 'article', 'source', 'topics']:
        for name in ['asset', 'plantpropertyequipment', 'annual_employees', 'sales']:
            if name=='annual_employees':
                y_multiplier = 1000  # correct units of annual_employees
            else:
                y_multiplier = 1
            
            df_ix = econdf[info_type].isna() | econdf[name].isna()
            x, y = econdf[info_type].loc[~df_ix].values, econdf[name].loc[~df_ix].values
            if info_type=='records':
                x0 = record_pow_fit['lb']
            elif info_type=='article':
                x0 = article_pow_fit['lb']
            elif info_type=='source':
                x0 = source_pow_fit['lb']
            elif info_type=='topics':
                x0 = topic_pow_fit['lb']
            fit_ix = (x>=x0) & (y>0)
            soln = log_reg(x[fit_ix], y[fit_ix] * y_multiplier)
            
            errs = np.log(y[fit_ix] * y_multiplier) - np.polyval(soln, np.log(x[fit_ix]))
            
            # create empty col
            econdf[f'err_{name}'] = np.nan + np.zeros(econdf.shape[0])
            
            # put in scaling fit deviation values into column
            ix = np.where(~df_ix.values)[0]
            ix = ix[fit_ix]
            ix_ = np.zeros(df_ix.shape[0], dtype=bool)
            ix_[ix] = True
            ix = ix_
            econdf.loc[ix, [f'err_{name}']] = errs

        econdf.to_parquet(f'cache/{info_type}_scaling_deviation.parquet')

def load_heaps():
    """Helper function."""
    if os.path.exists('cache/heaps_df.p'):
        with open('cache/heaps_df.p', 'rb') as f:
            bins, articledf, sourcedf, topicdf = list(pickle.load(f).values())
        return bins, articledf, sourcedf, topicdf

    # the following may take some time
    bins = np.unique(np.around(np.logspace(0, 8, 150)))

    articledf = heaps_scaling('article')
    articledf['log_bin_ix'] = np.digitize(articledf['domain_counts'], bins)-1

    sourcedf = heaps_scaling('source')
    sourcedf['log_bin_ix'] = np.digitize(sourcedf['domain_counts'], bins)-1

    topicdf = heaps_scaling('topics', iprint=True)
    topicdf['log_bin_ix'] = np.digitize(topicdf['domain_counts'], bins)-1

    save_pickle(['bins', 'articledf', 'sourcedf', 'topicdf'],
                'cache/heaps_df.p', True)
    return bins, articledf, sourcedf, topicdf

def heaps_fit():
    bins, articledf, sourcedf, topicdf = load_heaps()

    xplot = np.logspace(0, 7.9, 200)
    fitter = []

    # group only article_counts
    group = articledf.iloc[:,2:].groupby('log_bin_ix')
    counts_xy = group.size().reset_index().values
    m_xy = group.mean().reset_index().values
    max_xy = group.max().reset_index().values
    min_xy = group.min().reset_index().values

    all_ix = counts_xy[:,1]>=5
    min_ix = min_xy[:,1]>1

    # fit max line
    x = bins[max_xy[:,0].astype(int)[all_ix]]
    y = max_xy[:,1][all_ix]
    fitter.append(InfoFillHypothesis(x, y))

    # fit mean line
    x = bins[m_xy[:,0].astype(int)[all_ix&min_ix]]
    y = m_xy[:,1][all_ix&min_ix]
    fitter.append(InfoFillHypothesis(x, y))
    fitter[-1].fit_scaling_func(x, y,
                                bounds=[(-np.inf, np.log(fitter[-2].params[0])),
                                        (-np.inf, np.inf),
                                        (-np.inf, np.inf),
                                        (-np.inf, np.log(fitter[-2].params[-1]))])

    # group only source_counts
    group = sourcedf.iloc[:,2:].groupby('log_bin_ix')
    m_xy = group.mean().reset_index().values
    max_xy = group.max().reset_index().values
    min_xy = group.min().reset_index().values

    all_ix = counts_xy[:,1]>=5
    min_ix = min_xy[:,1]>1

    x = bins[max_xy[:,0].astype(int)[all_ix]]
    y = max_xy[:,1][all_ix]
    fitter.append(InfoFillHypothesis(x, y))

    x = bins[m_xy[:,0].astype(int)[all_ix&min_ix]]
    y = m_xy[:,1][all_ix&min_ix]
    fitter.append(InfoFillHypothesis(x, y))
    fitter[-1].fit_scaling_func(x, y,
                                bounds=[(-np.inf, np.log(fitter[-2].params[0])),
                                        (-np.inf, np.inf),
                                        (-np.inf, np.inf),
                                        (-np.inf, np.log(fitter[-2].params[-1]))])

    # group only topic_counts
    xmn = 10
    xmx = 1001
    group = topicdf.iloc[:,2:].groupby('log_bin_ix')
    counts_xy = group.size().reset_index().values
    m_xy = group.mean().reset_index().values
    max_xy = group.max().reset_index().values
    min_xy = group.min().reset_index().values

    all_ix = counts_xy[:,1]>=5
    min_ix = min_xy[:,1]>=10

    # fit max
    x = bins[max_xy[:,0][all_ix&min_ix].astype(int)]
    y = max_xy[:,1][all_ix&min_ix]/10
    fit_ix = (x>=xmn)&(x<=xmx)
    fitter.append(InfoFillHypothesis(x[fit_ix], y[fit_ix]))

    # only consider points far from cutoff
    x = bins[m_xy[:,0][all_ix&min_ix].astype(int)]
    y = m_xy[:,1][all_ix&min_ix]/10
    fit_ix = (x>=xmn)&(x<=xmx)
    fitter.append(InfoFillHypothesis(x[fit_ix], y[fit_ix]))
    fitter[-1].fit_scaling_func(x[fit_ix], y[fit_ix])

    save_pickle(['fitter'], 'cache/heaps_fit.p', True)

def heaps_fit_sensitivity():
    bins, articledf, sourcedf, topicdf = load_heaps()

    def opt_fit_params(x_mean, y_mean, x_max, y_max,
                    lower_range=range(15),
                    upper_range=np.logspace(0,3,20)):
        """Information coordination model fit sensitivity analysis. Try all combos of lower end and upp
        end that are given. Lower end is an index and the upper end is a percentage to make it
        consistent between the fits to different info values.
        
        Parameters
        ----------
        x_mean : ndarray
        y_mean : ndarray
        x_max : ndarray
        y_max : ndarray
        lower_range : list, range(15)
        upper_range : list, np.logspace(0, 3, 20)
        
        Returns
        -------
        twople
            Array index where fit error is minimized.
        ndarray
            Normalized fit error.
        """
        err_mean = np.zeros((len(lower_range), len(upper_range)))
        err_max = np.zeros((len(lower_range), len(upper_range)))
        a_mean = np.zeros((len(lower_range), len(upper_range)))
        b_mean = np.zeros((len(lower_range), len(upper_range)))
        a_max = np.zeros((len(lower_range), len(upper_range)))
        b_max = np.zeros((len(lower_range), len(upper_range)))
        
        for i, el in enumerate(lower_range):
            for j, hi in enumerate(upper_range):
                # fit max first b/c it establishes constrain for mean
                fit_ix = (x_max >= el) & (x_max <= hi)
                fit = InfoFillHypothesis(x_max[fit_ix], y_max[fit_ix]/10)
                err_max[i, j] = np.linalg.norm(np.log(fit.f(x_max)*10) - np.log(y_max)) / x_max.size
                a_max[i,j] = fit.a
                b_max[i,j] = fit.b

                fit_ix = (x_mean >= el) & (x_mean <= hi)
                fit_mean = InfoFillHypothesis(x_mean[fit_ix], y_mean[fit_ix]/10)
                fit_mean.fit_scaling_func(x_mean[fit_ix], y_mean[fit_ix]/10,
                                         initial_guess=fit.params)
                err_mean[i, j] = np.linalg.norm(np.log(fit_mean.f(x_mean)*10) - np.log(y_mean)) / x_mean.size
                a_mean[i,j] = fit_mean.a
                b_mean[i,j] = fit_mean.b
                
        return (np.unravel_index(np.argmin(err_mean), err_mean.shape), err_mean,
                np.unravel_index(np.argmin(err_max), err_max.shape), err_max,
                a_mean, a_max, b_mean, b_max)
    lower_range = np.arange(1, 11)
    upper_range = np.unique(np.floor(np.logspace(np.log10(30), 4, 50)))

    group = topicdf.iloc[:,2:].groupby('log_bin_ix')
    counts_xy = group.size().reset_index().values
    m_xy = group.mean().reset_index().values
    max_xy = group.max().reset_index().values
    min_xy = group.min().reset_index().values

    all_ix = counts_xy[:,1]>=5
    min_ix = min_xy[:,1]>=10

    output = opt_fit_params(bins[m_xy[:,0].astype(int)][all_ix&min_ix], m_xy[:,1][all_ix&min_ix],
                            bins[max_xy[:,0].astype(int)][all_ix&min_ix], max_xy[:,1][all_ix&min_ix],
                            lower_range=lower_range,
                            upper_range=upper_range)
    m_opt_ix, m_err, mx_opt_ix, mx_err, a_mean, a_max, b_mean, b_max = output
    fit_criterion_ix = ((a_max-a_mean)>=0) & (m_err<.1)

    save_pickle(['m_opt_ix', 'm_err', 'mx_opt_ix', 'mx_err', 'a_mean', 'a_max', 'b_mean', 'b_max',
                'lower_range', 'upper_range','fit_criterion_ix'],
                'cache/heaps_fitting_variation.p', True)