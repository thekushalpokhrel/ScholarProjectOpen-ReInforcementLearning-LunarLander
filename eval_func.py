# this is the evaluation function that is defined
def evaluate_model(model, X_ts, y_obs):
    y_pred = model.predict(X_ts)
    y_pred = np.argmax(y_pred, axis=1)

    tp = sum((y_pred == 1) & (y_obs == 1))
    fp = sum((y_pred == 1) & (y_obs == 0))
    tn = sum((y_pred == 0) & (y_obs == 0))
    fn = sum((y_pred == 0) & (y_obs == 1))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return accuracy, precision, recall
