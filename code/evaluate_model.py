from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(model):
  test_metrics = []
  test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=1)
  test_metrics.append({'Test Loss': test_loss,
                       'Test Accuracy': test_accuracy,
                       'Test Precision': test_precision,
                       'Test Recall': test_recall})
  return test_metrics
