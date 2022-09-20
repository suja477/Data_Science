def replacer(A):
    Q= A.isna().sum()
    missing_columns= list(Q[Q>0].index)
    for i in missing_columns:
        if( A[i].dtypes == "object"):
            x=A[i].mode()[0]
            A[i]=A[i].fillna(x)
        else:
            x=A[i].mean()
            A[i]=A[i].fillna(x)
            
            
def mean(T):
    sm = 0
    ct = 0
    for i in T:
        sm = sm + i
        ct = ct + 1
    mn = round(sm/ct,2)
    return mn
