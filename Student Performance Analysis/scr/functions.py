import numpy as np
from scipy.stats import shapiro
from scipy.stats import f
from scipy.stats import t

# Fuction for a F-test to compare the variances of two samples
def var_test(x, y, alpha):
    n1 = len(x)
    n2 = len(y)

    S21 = x.var(ddof=1)
    S22 = y.var(ddof=1)

    if S21 > S22:
        F_obs = S21/S22
        df1 = n1 - 1
        df2 = n2 - 2 
    elif S21 < S22:
        F_obs = S22/S21 
        df1 = n2 - 1
        df2 = n1 - 2
       
    f2 = f.ppf(1-alpha/2, df1, df2) 
    f1 = 1/(f.ppf(1-alpha/2, df2, df1))  
        
    if F_obs < f1 or F_obs > f2: 
        return "Reject H0: The variances are not equal"
    else:
        return "Fail to reject H0: The variances are equal"
    
# Function for a independent two-sample t-test 
def ind_mean_varequal(x, y, alpha, bi):
    
    if bi == "bilateral":
        n1 = len(x)
        n2 = len(y)
        S21 = x.var(ddof=1)
        S22 = y.var(ddof=1)

        Xmean = x.mean()
        Ymean = y.mean()

        S2xy = ((n1 - 1)*S21 + (n2 - 1)*S22)/(n1 + n2 - 2)
        
        tc1 = t.ppf(alpha/2, n1 + n2 - 2)
        tc2 = t.ppf(1 - alpha/2, n1 + n2 - 2)
        
        T_obs = (Xmean - Ymean) / (np.sqrt(S2xy * (1/n1 + 1/n2)))
        if T_obs < tc1 or T_obs > tc2: 
            return f"Reject H0: T_obs = {T_obs:.3f}, tc1 = {tc1:.3f}, tc2 = {tc2:.3f}"
        else:
            return f"Fail to reject H0: T_obs = {T_obs:.3f}, tc1 = {tc1:.3f}, tc2 = {tc2:.3f}"
    
    elif bi == "left-tailed": 
        n1 = len(x)
        n2 = len(y)
        S21 = x.var(ddof=1)
        S22 = y.var(ddof=1)

        Xmean = x.mean()
        Ymean = y.mean()

        S2xy = ((n1 - 1)*S21 + (n2 - 1)*S22)/(n1 + n2 - 2)
        
        tc = t.ppf(1 - alpha, n1 + n2 - 2)
        
        T_obs = (Xmean - Ymean) / (np.sqrt(S2xy * (1/n1 + 1/n2)))
        if T_obs < tc: 
            return f"Reject H0: T_obs = {T_obs:.3f}, tc = {tc:.3f}"
        else:
            return f"Fail to reject H0: T_obs = {T_obs:.3f}, tc = {tc:.3f}"
    
    elif bi == "right-tailed": 
        n1 = len(x)
        n2 = len(y)
        S21 = x.var(ddof=1)
        S22 = y.var(ddof=1)

        Xmean = x.mean()
        Ymean = y.mean()

        S2xy = ((n1 - 1)*S21 + (n2 - 1)*S22)/(n1 + n2 - 2)
        
        tc = t.ppf(alpha, n1 + n2 - 2)
        
        T_obs = (Xmean - Ymean) / (np.sqrt(S2xy * (1/n1 + 1/n2)))
        if T_obs > tc: 
            return f"Reject H0: T_obs = {T_obs:.3f}, tc = {tc:.3f}"
        else:
            return f"Fail to reject H0: T_obs = {T_obs:.3f}, tc = {tc:.3f}"

# Confidence interval for the difderence between two means 

def ic(x, y, alpha):
    n1 = len(x)
    n2 = len(y)
    S21 = x.var(ddof=1)
    S22 = y.var(ddof=1)

    Xmean = x.mean()
    Ymean = y.mean()

    S2xy = ((n1 - 1)*S21 + (n2 - 1)*S22)/(n1 + n2 - 2)

    t_crit = t.ppf(1 - alpha/2, n1 + n2 - 2)
    
    margin_error = t_crit * np.sqrt(S2xy * (1/n1 + 1/n2))
    
    lower_bound = (Xmean - Ymean) - margin_error
    upper_bound = (Xmean - Ymean) + margin_error
    
    return lower_bound, upper_bound

# Confidence interval for the correlation between two variables using bootstrap

def bootstrap_correlation(x, y, b_iterations, alpha):
    corre_values = []
    n = len(x)
    
        # We must resample x and y as a single sample from the same student — only this way does the correlation make sense
        # In other words, if a student has 5 hours of social media usage, they should still have 5 hours in the resampled data,
        # and their grade should remain the same in the resampled data
        # Therefore, we will resample the indices of x and y together
        # This ensures that the resampled data maintains the relationship between the two variables

    
    for _ in range(b_iterations):
        indices = np.random.choice(range(n), size=n, replace=True)
        # We are selecting the same random observations for both x and y using the indices
        x_sample = x.iloc[indices]
        y_sample = y.iloc[indices]
        
        corr = x_sample.corr(y_sample)
        corre_values.append(corr)
    
    corre_std = np.std(corre_values)
    lower_bound = np.percentile(corre_values, 100 * alpha / 2)
    upper_bound = np.percentile(corre_values, 100 * (1 - alpha / 2))
    return lower_bound, upper_bound, corre_std