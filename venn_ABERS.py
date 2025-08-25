import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.calibration import CalibrationDisplay
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.calibration import CalibratedClassifierCV

import warnings
warnings.filterwarnings('ignore')



from venn_abers import VennAbersCalibrator, VennAbers



# create binary classification dataset
np.random.seed(seed=1)

X, y = make_classification(
    n_samples=100000, n_features=20, n_informative=2, n_redundant=2, random_state=1
)

train_samples = 1000  # Samples used for training the models
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    shuffle=False,
    test_size=100000 - train_samples,
)

# applying ven_ABERS calibration to an underlying classifier
# underlying classifier
clf = GaussianNB()
clf.fit(X_train, y_train)
clf_prob = clf.predict_proba(X_test)

# Inductive Venn-ABERS calibration (IVAP)
va = VennAbersCalibrator(estimator=clf, inductive=True, cal_size=0.2, shuffle=False)
va.fit(X_train, y_train)
va_inductive_prob = va.predict_proba(X_test)

# Cross Venn-ABERS calibration (CVAP)
va = VennAbersCalibrator(estimator=clf, inductive=False, n_splits=2)
va.fit(X_train, y_train)
va_cv_prob = va.predict_proba(X_test)

# Pre-fitted Venn-ABERS calibration
X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(
    X_train, y_train, test_size=0.2, shuffle=False
)

clf.fit(X_train_proper, y_train_proper)
p_cal = clf.predict_proba(X_cal)
p_test = clf.predict_proba(X_test)

va = VennAbersCalibrator()
va_prefit_prob = va.predict_proba(p_cal=p_cal, y_cal=y_cal, p_test=p_test)



log_losses=[]
log_losses.append(log_loss(y_test, clf_prob))
log_losses.append(log_loss(y_test, va_inductive_prob))
log_losses.append(log_loss(y_test, va_cv_prob))
log_losses.append(log_loss(y_test, va_prefit_prob))


df_loss = pd.DataFrame(columns=['Metric', 'Uncalibrated', 'IVAP', 'CVAP', 'Prefit'])
df_loss.loc[1] = ['log loss'] + log_losses
df_loss.set_index('Metric', inplace=True)
df_loss.round(3)
print(df_loss)