import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , ConfusionMatrixDisplay, accuracy_score, classification_report, precision_score, recall_score


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

'''
a) Prikažite podatke za ucenje u x1 x2 ravnini matplotlib biblioteke pri cemu podatke obojite
s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
marker (npr. x). Koristite funkciju scatter koja osim podataka prima i parametre c i
cmap kojima je moguce definirati boju svake klase.
'''
plt.scatter(X_train[:,0], X_train[:,1], cmap="coolwarm", label="Train data", alpha=0.7, c=y_train)
plt.scatter(X_test[:,0], X_test[:,1], cmap="coolwarm", marker="*", label="Test data", alpha=0.7, c=y_test)
plt.xlabel("x1")
plt.ylabel("x2")
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
c) Pronadite u atributima izgrađenog modela parametre modela. Prikažite granicu odluke 
naucenog modela u ravnini  x1  x2 zajedno s podacima za ucenje. Napomena: granica 
odluke u ravnini x1 x2 definirana je kao krivulja: θ0 +θ1x1 +θ2x2 = 0.
'''
coef = LogRegression_model.coef_[0]
intercept = LogRegression_model.intercept_


def decision_boundary(x1):
    return (-coef[0]*x1 - intercept) / coef[1]

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm')
plt.plot(X_train[:, 0], decision_boundary(X_train[:, 0]))
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Logistic Regression Decision Boundary')
plt.show()


'''
d) Provedite klasifikaciju skupa podataka za testiranje pomocu izgrađenog modela logističke ˇ
regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izračunate točnost, ˇ
preciznost i odziv na skupu podataka za testiranje.
'''
disp = ConfusionMatrixDisplay( confusion_matrix ( y_test , y_test_p ) )
disp.plot()
plt.show()

print (" Tocnost : " ,accuracy_score( y_test , y_test_p ))
print (" Preciznost : " ,precision_score( y_test , y_test_p ))
print (" Odziv : " ,recall_score( y_test , y_test_p ))
print(classification_report( y_test , y_test_p ) )


'''
e) Prikažite skup za testiranje u ravnini x1 x2. Zelenom bojom oznacite dobro klasificirane
primjere dok pogrešno klasificirane primjere oznacite crnom bojom.
'''
plt.scatter(X_test[:, 0], X_test[:, 1], label="test", cmap="seismic", c=y_test)
for i in range(len(y_test)):
    if y_test[i] == y_test_p[i]:
        plt.scatter(X_test[i, 0], X_test[i, 1], c='g')
    else:
        plt.scatter(X_test[i, 0], X_test[i, 1], c='r')

plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

