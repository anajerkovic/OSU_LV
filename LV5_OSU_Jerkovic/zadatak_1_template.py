import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay, accuracy_score, classification_report


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

'''
a) Prikažite podatke za ucenje u x1 −x2 ravnini matplotlib biblioteke pri cemu podatke obojite
s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
marker (npr. ’x’). Koristite funkciju scatter koja osim podataka prima i parametre c i
cmap kojima je moguce de ´ finirati boju svake klase.
'''
plt.scatter(X_train[:,0], X_train[:,1], c="magenta", label="Train data", alpha=0.5)
plt.scatter(X_test[:,0], X_test[:,1], c="green", marker="*", label="Test data", alpha=0.5)
plt.legend()
plt.show()


'''
b) Izgradite model logisticke regresije pomoću scikit-learn biblioteke na temelju skupa poda-
taka za ucenje.
'''
LogRegression_model = LogisticRegression()
LogRegression_model.fit( X_train , y_train )
y_test_p = LogRegression_model.predict( X_test )


'''
c) Pronadite u atributima izgrađenog modela parametre modela. Prikažite granicu odluke ¯
naucenog modela u ravnini ˇ x1 − x2 zajedno s podacima za ucenje. Napomena: granica ˇ
odluke u ravnini x1 −x2 definirana je kao krivulja: θ0 +θ1x1 +θ2x2 = 0.
'''
b = LogRegression_model.intercept_[0] # θ0
w1,w2 = LogRegression_model.coef_.T  # θ1 θ2

c = -b/w2
m = -w1/w2 # nagib

xd = np.array([-4, 4])
yd = m*xd+c

plt.plot(xd,yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, -4,color='magenta', alpha=0.3)
plt.fill_between(xd, yd, 4, color='green', alpha=0.3)
plt.show()


'''
d) Provedite klasifikaciju skupa podataka za testiranje pomocu izgrađenog modela logističke ˇ
regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izračunate točnost, ˇ
preciznost i odziv na skupu podataka za testiranje.
'''
disp = ConfusionMatrixDisplay( confusion_matrix ( y_test , y_test_p ) )
disp.plot ()
plt.show ()
print (" Tocnost : " ,accuracy_score( y_test , y_test_p ))
print(classification_report( y_test , y_test_p ) )


'''
e) Prikažite skup za testiranje u ravnini x1 −x2. Zelenom bojom oznacite dobro klasificirane
primjere dok pogrešno klasificirane primjere oznacite crnom bojom.
'''
