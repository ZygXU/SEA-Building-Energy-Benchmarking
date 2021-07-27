import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



# This fonction allows to print with histograms the pourcentage of data available in dataset,
# by columns or by rows (depending parameters).
# 0 if you want print by columns, and 1 if you want print by rows.
def print_row_col(row_col,df):
    fig = plt.figure()
    if(row_col==1):
        #for each row, we have ...
        pourcentage=1-(df.isna().sum(axis=1)/df.shape[1])
        plt.hist(pourcentage, bins=df.shape[1])
        plt.title('Pourcentage of data available by products (each rows) ')
    else:
        #for each characteristics, we have ...
        pourcentage=1-(df.isna().sum()/df.shape[0])
        plt.hist(pourcentage)
        plt.title('Pourcentage of data available by characteristics (each columns)  ')
    
    return pourcentage



def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.


    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh



#fonction print ETA² => ANOVA Rapport de correlation Pearson
#Proche de 0 : Moyenne par classe égale =>pas de relation
#Entre 0 et 1 : Moyenne par classe différent => relation
# x :qualitative
# y: quantitative
def eta_squared(x,y):
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y]) #variation totale (sum des carrée totale)
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes]) #variation interclasse
    return SCE/SCT



#Fonction CrossValidation stratified
#Fonction qui split en stratified sur une variable précis en X
#s'utilise dans les cas ou les données ne sont pas homogenes en X
#X,y entré dataset pour la CV
#stratified_feature : le feature non homogene à stratified
def stratified_split_X(X,y,stratified_feature,model,n_split=3):

    scores=[]
    for i in range(n_split):
        sss_X_train, sss_X_test, sss_y_train1, sss_y_test1 = train_test_split(X, y, test_size=1/n_split, random_state=i, 
              stratify= stratified_feature)

        model.fit(sss_X_train, sss_y_train1) 
        #pred = model.predict(sss_X_test)
        #print(sss_y_test1.values)
        #print(pred)
        scores.append(model.score(sss_X_test, sss_y_test1))
    
    return scores


