import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import pickle
import matplotlib.pyplot as plt
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a SVM model for wine quality prediction')
    parser.add_argument('--data_path', type=str, default='./winequality.csv',
                        help='Path to the wine quality data file')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of items in the test set')
    parser.add_argument('--features', type=str, nargs='+',
                        default=None, help='Features used for training')
    parser.add_argument('--kernel', type=str, default='rbf',
                        help='Kernel type of the SVM model')
    parser.add_argument('--result_path', type=str,
                        default='./results', help='Location to save results')
    return parser.parse_args()


def get_RMSE(y_test, predictions):
    return mean_squared_error(y_test, predictions) ** 0.5


def plot_RMSE(y_test, predictions, rmse, r2, result_path):
    plt.scatter(y_test, predictions)
    plt.xlabel('True Quality')
    plt.ylabel('Predicted Quality')
    plt.title(f'SVM Model Results\nRMSE: {rmse:.2f}, R2 Score: {r2:.2f}')
    plt.savefig(f'{result_path}/RMSE.png')


def plot_RESIDUALS(residuals, result_path):
    plt.hist(residuals, bins=20, edgecolor='black')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Histogram of Residuals')
    plt.savefig(f'{result_path}/Residuals.png')


def plot_cv_scores(cv_scores, result_path):
    plt.plot(cv_scores, color='blue', marker='o', markersize=5)
    plt.xlabel('Cross-Validation Fold')
    plt.ylabel('R2 Score')
    plt.title('Cross-Validation Scores')
    plt.savefig(f'{result_path}/cv_scores.png')


def save_data(results, result_path):
    with open(f'{result_path}/results.pkl', 'wb') as f:
        pickle.dump(results, f)

    json_object = json.dumps(results, indent=4)
    with open(f"{result_path}/results.json", "w") as outfile:
        outfile.write(json_object)


def main():
    args = parse_args()
    df = pd.read_csv(args.data_path)
    result_path = args.result_path

    if args.features:
        X = df[args.features]
    else:
        X = df.drop(columns='quality')

    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42)

    model = SVR(kernel=args.kernel)
    scores = cross_val_score(model, X, y, cv=10)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = get_RMSE(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    residuals = y_test - predictions

    plot_RMSE(y_test, predictions, rmse, r2, result_path)
    plot_RESIDUALS(residuals, result_path)
    plot_cv_scores(scores, result_path)

    results = {'true_values': y_test.to_json(), 'predictions': predictions.tolist(), "rmse": rmse, "r2": r2, "residuals": residuals.to_json(),
               "cross_scores": scores.tolist(), "cross_mean_score": scores.mean()}
    save_data(results, result_path)

    print('Model training completed successfully!')
    print('Results saved to:', result_path)


if __name__ == '__main__':
    main()
