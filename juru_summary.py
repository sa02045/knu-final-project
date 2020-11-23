import pandas as pd

def year_sample(year):
    from_dir = 'juru (1)/'
    path = from_dir+year+'_juru_sample.csv'
    df=pd.read_csv(path)
    df = df.dropna()
    for i in ['상대', '타자', '도루', '도실', '도루기회', '도루시도%']:
        df = df.drop(i, axis=1)
    df_d_d = df.drop_duplicates(['도루RAA', '도루 성공', '도루 실패', '도실%'], keep='first')

    return df_d_d

def juru_summary(years):
    df = pd.DataFrame({
        "도루RAA" : [],
        "도루 성공" : [],
        "도루 실패" : [],
        '도실%' : [],}
        )
    # print(df)

    for year in years :
        df = pd.concat([df,year_sample(year)])

    return df

if __name__ == '__main__':
    years = ['2016', '2017', '2018', '2019', '2020']