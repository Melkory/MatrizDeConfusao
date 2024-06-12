import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Suponhamos que estas sejam as classificações reais e previstas
y_true = np.array([0, 1, 0, 1, 0, 1, 1, 0, 0, 1])
y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1])

# Calculando a matriz de confusão
cm = confusion_matrix(y_true, y_pred)

# Calculando as métricas
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Calculando especificidade manualmente
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

# Exibindo os resultados
print(f"Matriz de Confusão:\n{cm}")
print(f"Acurácia: {accuracy}")
print(f"Precisão: {precision}")
print(f"Sensibilidade (Recall): {recall}")
print(f"Especificidade: {specificity}")
print(f"F-score: {f1}")