def catcon(df):
    cat = []
    con = []
    for i in df.columns:
        if(df[i].dtypes == "object"):
            cat.append(i)
        else:
            con.append(i)
    return cat,con
    
def OL(df):
    out = []
    cat,con = catcon(df)
    df = standardize(df)
    for i in con:
        out.extend(list(df[(df[i]>3)|(df[i]<-3)].index))

    import numpy as np
    outliers = list(np.unique(out))
    return outliers
    
def standardize(df):
    cat,con = catcon(df)
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    import pandas as pd
    Q = pd.DataFrame(ss.fit_transform(df[con]),columns=con)
    return Q
    
def replacer(df):
    cat,con = catcon(df)
    for i in cat:
        x=df[i].mode()[0]
        df[i]=df[i].fillna(x)

    for i in con:
        x=df[i].mean()
        df[i]=df[i].fillna(x)