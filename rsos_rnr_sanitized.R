require(arrow)
require(RClickhouse)
require(data.table)
require(tidyverse)
require(fixest)

querySQLDatabase=function(x){
  
  dbGetQuery(con,x) %>% as.data.table
}


setwd(dropbox_path()); setwd('datafolder/')
'rsos_data/sales_scaling_deviation.parquet' %>% read_parquet -> sales_scaling_deviation
'rsos_data/records_scaling_deviation.parquet'  %>% read_parquet -> records_scaling_deviation
'rsos_data/topics_scaling_deviation.parquet' %>% read_parquet -> topics_scaling_deviation
'rsos_data/source_scaling_deviation.parquet' %>% read_parquet -> source_scaling_deviation
'rsos_data/article_scaling_deviation.parquet' %>% read_parquet -> article_scaling_deviation

sales_scaling_deviation
 
records_scaling_deviation$domain %>% n_distinct()
if(0){
  sa=querySQLDatabase("
        select distinct * from 
          (select tobinsq,toInt32(company.gvkey) as gvkey,datadate,cutToFirstSignificantSubdomain(weburl) as domain 
          from comp.standard_annual
            inner join 
              (
              select distinct * from (select weburl,toInt32(gvkey) as gvkey from comp.company
              where weburl is not null and weburl!=''
                  union all 
                  select weburl,toInt32(gvkey) as gvkey from comp.namesfile_snapshot
                    where enddate>='2019-01-01' and begdate<='2020-01-01'
                    and weburl is not null and weburl!=''
                  )
              ) as company
           on toInt32(standard_annual.gvkey)=toInt32(company.gvkey)) as a 
           inner join (select gvkey,exp(sum(ln(ret+1)))-1 as ret from
            crsp.msf inner join link.ccmxpf_linktable
              on lpermno=permno and usedflag=1
              where toString(date) between linkdt and coalesce(linkenddt, '2099-12-31')
                and date>='2019-01-01'
              group by gvkey) as b
           on a.gvkey=toInt32(b.gvkey)
           ")
  # segments
  {
    segments=querySQLDatabase("select * from (select *,max(datadate) over(partition by gvkey) as latest from comp_202401.wrds_segmerged
            where year(date(datadate))=2018)
            where datadate=latest
            ")
    segments[,num_sic:=n_distinct(sics1),by=list(gvkey)]
    segments[,sales_in_sic:=revts %>% sum(na.rm=TRUE),by=list(gvkey,sics1)]
    segments[,list(sales_in_sic,gvkey,sics1)] %>% unique %>%
      .[,list(diversity_sales=1-hhi(sales_in_sic),
             num_sic=n_distinct(sics1)
              ),by=list(gvkey)] ->segment_diversity
  }
  # innovation
  {
    queryClickhouseDb("select * from  patents.patentview_uspc_current limit 5
                   ")
    res=queryTailscale("select gvkey,mainclass_id,count() as num_patent from 
                     patents.patentview_uspc_current
                     inner join patents.kpss_2022
                    on toInt32(patent_id)=toInt32(patent_num)
                    inner join link.ccmxpf_linktable
                      on toInt32(lpermno)=toInt32(permno) and usedflag=1
                    where filing_date2>='2010-01-01'
                    and filing_date2 between linkdt and coalesce(linkenddt, '2099-12-31')
                    group by all
                   ")
    res=res[,list(diversity_ip=1-hhi(num_patent),patents=sum(num_patent)),by=list(gvkey=as.integer(gvkey))]
  }
  # Product market fluidity
  
  
  res[,gvkey:=as.integer(gvkey)]
  
  setwd(dropbox_path()); setwd('datafolder/')
  'rsos_data/records_scaling_deviation.parquet'  %>% read_parquet -> records_scaling_deviation
  'rsos_data/topics_scaling_deviation.parquet' %>% read_parquet -> topics_scaling_deviation
  'rsos_data/source_scaling_deviation.parquet' %>% read_parquet -> source_scaling_deviation
  'rsos_data/article_scaling_deviation.parquet' %>% read_parquet -> article_scaling_deviation
  
  for(wildcard in c('records','topics','source','article')){
    if(wildcard=='records') readvar=records_scaling_deviation
    if(wildcard=='topics') readvar=topics_scaling_deviation
    if(wildcard=='source') readvar=source_scaling_deviation
    if(wildcard=='article') readvar=article_scaling_deviation
    
    d2=merge(sa[year(datadate)==2018,],readvar,by='domain',all.y=TRUE) %>% getLast(c('domain')) %>%
        merge(.,segment_diversity[,gvkey:=as.integer(gvkey)],all.x=TRUE,by='gvkey') %>%
        merge(.,res,all.x=TRUE,by='gvkey')
    colnames(records_scaling_deviation)
    
    write_parquet(d2,'~/dropboxpathdata/alan_dataset_correlate_to_outcomes_rsos_{wildcard}.parquet' %>% str_glue)
  }
  
  queryClickhouseDb("select toInt32(gvkey) as gvkey,naics from comp.company") %>% write_parquet('~/dropboxpathdata/naics_gvkey.parquet')
}
changed_files=base::array()[0]
read_parquet('dropboxpath/data/naics_gvkey.parquet') %>% as.data.table -> naics_gvkey
for(wildcard in c('records','source','article')){
  d2=read_parquet('~/dbxmount//collab_omg_its_eddie_lee/data/alan_dataset_correlate_to_outcomes_rsos_{wildcard}.parquet' %>% str_glue) %>% as.data.table
  d2=merge(d2,naics_gvkey,by='gvkey')
  w=function(x) winsor(x) %>% zscore
  
  print('###################################')
  print('Wildcard {wildcard}' %>% str_glue)
  print('###################################')
  
  w=function(x) winsor(x) %>% zscore
  d2[,naics2:=substr(naics,1,2)]
  wflip=function(x) w(x*(-1))
  colnames(d2) %>% grepv('plant')
  suppressWarnings({ 
    for(mode in c('asset','sales','plantpropertyequipment','annual_employees')){
      l=list()
      wflip=function(x) w(x*(-1))
      l[[3]]=feols(data=d2,fml='w(diversity_sales) ~wflip(err_{mode})+log({mode})' %>% str_glue %>% as.formula,warn = FALSE) 
      l[[4]]=feols(data=d2,fml='w(diversity_sales) ~wflip(err_{mode})+log({mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
      l[[1]]=feols(data=d2,fml='w(diversity_ip) ~wflip(err_{mode})+log({mode})' %>% str_glue %>% as.formula,warn = FALSE)
      l[[2]]=feols(data=d2,fml='w(diversity_ip) ~wflip(err_{mode})+log({mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
      l[[5]]=feols(data=d2,fml='w(tobinsq) ~wflip(err_{mode})+log({mode})' %>% str_glue %>% as.formula,warn = FALSE)
      l[[6]]=feols(data=d2,fml='w(tobinsq) ~wflip(err_{mode})+log({mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
      # ret
      l[[7]]=feols(data=d2,fml='winsor(ret) ~wflip(err_{mode})+log({mode})' %>% str_glue %>% as.formula,warn = FALSE)
      l[[8]]=feols(data=d2,fml='winsor(ret) ~wflip(err_{mode})+log({mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
      
      dict2=c('wflip(err_asset)'='Excess Reading wrt Assset','log(asset)'='Log asset',
              'wflip(err_sales)'='Excess Reading wrt Sales','log(sales)'='Log sales',
              'wflip(err_plantpropertyequipment)'='Excess Reading wrt PPE','log(plantpropertyequipment)'='Log PPE',
              'wflip(err_annual_employees)'='Excess Reading wrt Employees','log(annual_employees)'='Log Employees',
              'w(diversity_ip)'='IP Diversity','w(diversity_sales)'='Sales Diversity',
              'w(tobinsq)'='Tobin Q','winsor(ret)'='Return')
      
      path='/home/alan/dbxmount//collab_omg_its_eddie_lee/apk/tables_rsos/'
      etable(l,se.below=TRUE,signif.code=c('***'=0.01,'**'=0.05,'*'=0.1),cluster='gvkey',dict=dict2) %>% print
      name='regtable_{mode}_{wildcard}.tex' %>%str_glue
      etable(l,se.below=TRUE,signif.code=c('***'=0.01,'**'=0.05,'*'=0.1),cluster='gvkey',tex=TRUE,style.tex=style.tex('aer'),
             dict=dict2  
      ) %>%
        writeLines(path:name)
      
      changed_files=c(changed_files,path:name)
      
    }
  })
  
}
##### ignore whats belo where
 

suppressWarnings({
  w=function(x) winsor(x) %>% zscore
  d2[,naics2:=substr(naics,1,2)]
  colnames(d2) %>% grepv('plant')
  suppressWarnings({ 
    for(mode in c('asset','sales','plantpropertyequipment','annual_employees')){
      l=list()
      l[[1]]=feols(data=d2,fml='winsor(diversity_sales) ~w(log_records)+log({mode})' %>% str_glue %>% as.formula,warn = FALSE) 
      l[[2]]=feols(data=d2,fml='winsor(diversity_sales) ~w(log_records)+log({mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
      l[[3]]=feols(data=d2,fml='winsor(diversity_ip) ~w(log_records)+log({mode})' %>% str_glue %>% as.formula,warn = FALSE)
      l[[4]]=feols(data=d2,fml='winsor(diversity_ip) ~w(log_records)+log({mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
      l[[5]]=feols(data=d2,fml='winsor(tobinsq) ~w(log_records)+log({mode})' %>% str_glue %>% as.formula,warn = FALSE)
      l[[6]]=feols(data=d2,fml='winsor(tobinsq) ~w(log_records)+log({mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
      # ret
      l[[7]]=feols(data=d2,fml='winsor(ret) ~w(log_records)+log({mode})' %>% str_glue %>% as.formula,warn = FALSE)
      l[[8]]=feols(data=d2,fml='winsor(ret) ~w(log_records)+log({mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
      
      
      
      etable(l,se.below=TRUE,signif.code=c('***'=0.01,'**'=0.05,'*'=0.1),cluster='gvkey') %>% print
      
    }
  })
})

w=function(x) winsor(x) %>% zscore
d2[,naics2:=substr(naics,1,2)]
colnames(d2) %>% grepv('plant')
suppressWarnings({ 
  for(mode in c('asset','sales','plantpropertyequipment','annual_employees')){
    l=list()
    wflip=function(x) w(x*(-1))
    l[[3]]=feols(data=d2,fml='winsor(diversity_sales) ~wflip(err_{mode})+log({mode})' %>% str_glue %>% as.formula,warn = FALSE) 
    l[[4]]=feols(data=d2,fml='winsor(diversity_sales) ~wflip(err_{mode})+log({mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
    l[[1]]=feols(data=d2,fml='winsor(diversity_ip) ~wflip(err_{mode})+log({mode})' %>% str_glue %>% as.formula,warn = FALSE)
    l[[2]]=feols(data=d2,fml='winsor(diversity_ip) ~wflip(err_{mode})+log({mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
    l[[5]]=feols(data=d2,fml='winsor(tobinsq) ~wflip(err_{mode})+log({mode})' %>% str_glue %>% as.formula,warn = FALSE)
    l[[6]]=feols(data=d2,fml='winsor(tobinsq) ~wflip(err_{mode})+log({mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
    # ret
    l[[7]]=feols(data=d2,fml='winsor(ret) ~wflip(err_{mode})+log({mode})' %>% str_glue %>% as.formula,warn = FALSE)
    l[[8]]=feols(data=d2,fml='winsor(ret) ~wflip(err_{mode})+log({mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
    
    dict2=c('wflip(err_asset)'='Excess Reading wrt Assset','log(asset)'='Log asset',
         'wflip(err_sales)'='Excess Reading wrt Sales','log(sales)'='Log sales',
         'wflip(err_plantpropertyequipment)'='Excess Reading wrt PPE','log(plantpropertyequipment)'='Log PPE',
         'wflip(err_annual_employees)'='Excess Reading wrt Employees','log(annual_employees)'='Log Employees')
    
    path='/sabrent4/dropbox/Command Center Dropbox/Alan Kwan/collab_omg_its_eddie_lee/apk/tables_rsos/'
    etable(l,se.below=TRUE,signif.code=c('***'=0.01,'**'=0.05,'*'=0.1),cluster='gvkey',dict=dict2) %>% print
    name='regtable_{mode}.tex' %>%str_glue
    etable(l,se.below=TRUE,signif.code=c('***'=0.01,'**'=0.05,'*'=0.1),cluster='gvkey',tex=TRUE,style.tex=style.tex('aer'),
           dict=dict2  
           ) %>%
        writeLines(path:name)
           
    
  }
})


w=function(x) winsor(x) %>% zscore
d2[,naics2:=substr(naics,1,2)]
colnames(d2) %>% grepv('plant')
suppressWarnings({ 
  for(mode in c('asset','sales','plantpropertyequipment','annual_employees')){
    l=list()
    l[[1]]=feols(data=d2[get(mode)>0,],fml='winsor(diversity_sales) ~w(log_records/{mode})' %>% str_glue %>% as.formula,warn = FALSE) 
    l[[2]]=feols(data=d2[get(mode)>0,],fml='winsor(diversity_sales) ~w(log_records/{mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
    l[[3]]=feols(data=d2[get(mode)>0,],fml='winsor(diversity_ip) ~w(log_records/{mode})' %>% str_glue %>% as.formula,warn = FALSE)
    l[[4]]=feols(data=d2[get(mode)>0,],fml='winsor(diversity_ip) ~w(log_records/{mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
    l[[5]]=feols(data=d2[get(mode)>0,],fml='winsor(tobinsq) ~w(log_records/{mode})' %>% str_glue %>% as.formula,warn = FALSE)
    l[[6]]=feols(data=d2[get(mode)>0,],fml='winsor(tobinsq) ~w(log_records/{mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
    # ret
    l[[7]]=feols(data=d2[get(mode)>0,],fml='winsor(ret) ~w(log_records/{mode})' %>% str_glue %>% as.formula,warn = FALSE)
    l[[8]]=feols(data=d2[get(mode)>0,],fml='winsor(ret) ~w(log_records/{mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
    
    
    
    etable(l,se.below=TRUE,signif.code=c('***'=0.01,'**'=0.05,'*'=0.1),cluster='gvkey') %>% print
    
  }
})

suppressWarnings({ 
  for(mode in c('asset','sales','plantpropertyequipment','annual_employees')){
    l=list()
    l[[1]]=feols(data=d2,fml='winsor(diversity_sales) ~w(err_{mode})' %>% str_glue %>% as.formula,warn = FALSE) 
    l[[2]]=feols(data=d2,fml='winsor(diversity_sales) ~w(err_{mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
    l[[3]]=feols(data=d2,fml='winsor(diversity_ip) ~w(err_{mode})' %>% str_glue %>% as.formula,warn = FALSE)
    l[[4]]=feols(data=d2,fml='winsor(diversity_ip) ~w(err_{mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
    l[[5]]=feols(data=d2,fml='winsor(tobinsq) ~w(err_{mode})' %>% str_glue %>% as.formula,warn = FALSE)
    l[[6]]=feols(data=d2,fml='winsor(tobinsq) ~w(err_{mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
    # ret
    l[[7]]=feols(data=d2,fml='winsor(ret) ~w(err_{mode})' %>% str_glue %>% as.formula,warn = FALSE)
    l[[8]]=feols(data=d2,fml='winsor(ret) ~w(err_{mode}) |naics2' %>% str_glue %>% as.formula,warn = FALSE)
    
    
    
    etable(l,se.below=TRUE,signif.code=c('***'=0.01,'**'=0.05,'*'=0.1),cluster='gvkey') %>% print
    
  }
})
