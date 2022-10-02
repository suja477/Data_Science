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
        
def preprocessing(df):
    from Wd8pm import catcon,standardize
    import pandas as pd
    cat,con = catcon(df)
    X1 = standardize(df)
    X2 = pd.get_dummies(df[cat])
    Xnew = X1.join(X2)
    return Xnew

def regression(mob,xtrain,xtest,ytrain,ytest):
    from sklearn.linear_model import LinearRegression
    model = mob.fit(xtrain,ytrain)
    tr_pred = model.predict(xtrain)
    ts_pred = model.predict(xtest)
    from sklearn.metrics import mean_absolute_error,explained_variance_score
    adj = explained_variance_score(ytest,ts_pred)
    bias = mean_absolute_error(ytrain,tr_pred)
    var = mean_absolute_error(ytest,ts_pred)
    return adj,bias,var
        
def anova(A,B,df):
    from statsmodels.formula.api import ols
    model = ols(A +" ~ "+ B,df).fit()

    from statsmodels.stats.anova import anova_lm
    Q=anova_lm(model)
    return Q

def backward_elim(Xnew,Y):
    from sklearn.model_selection import train_test_split
    xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=41)
    from statsmodels.api import add_constant
    xconst=add_constant(xtrain)
    from statsmodels.api import OLS
    q=OLS(ytrain,xconst).fit()    
    return q