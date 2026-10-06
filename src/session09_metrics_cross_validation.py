"""CS156 Session 9: metrics, cross-validation, ROC-AUC, and data splitting."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn import datasets, linear_model, metrics, model_selection, svm


NUM_DATAPOINTS = 300
RANDOM_STATE = 42


def plot_decision_surface(clf, xs, ys, ax, title, x_data, y_data):
    """Plot P(class 1) over a 2D feature grid."""
    xs = np.linspace(xs[0], xs[1], 100)
    ys = np.linspace(ys[0], ys[1], 100)
    x_mesh, y_mesh = np.meshgrid(xs, ys)

    vis_x = np.stack((x_mesh, y_mesh), axis=2).reshape(-1, 2)
    vis_y = clf.predict_proba(vis_x)[:, 1].reshape(100, 100)

    contour = ax.contourf(
        x_mesh,
        y_mesh,
        vis_y,
        levels=np.linspace(0, 1, 33),
        cmap="Blues",
    )
    contour.set_clim(0, 1)

    ax.scatter(
        x_data[:, 0],
        x_data[:, 1],
        c=y_data,
        edgecolors="grey",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    ax.set_title(title)
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")


def tune_svm(x_train, y_train):
    """Choose gamma and C using an 80/20 validation split and ROC-AUC."""
    x_train_sub, x_val, y_train_sub, y_val = model_selection.train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    gamma_values = [0.1, 0.5, 1, 2, 5, 10]
    C_values = [0.1, 0.5, 1, 2, 5, 10]

    best_auc = -1.0
    best_gamma = None
    best_C = None

    for gamma in gamma_values:
        for C in C_values:
            model = svm.SVC(gamma=gamma, C=C, probability=True)
            model.fit(x_train_sub, y_train_sub)

            val_preds = model.predict_proba(x_val)[:, 1]
            val_auc = metrics.roc_auc_score(y_val, val_preds)

            if val_auc > best_auc:
                best_auc = val_auc
                best_gamma = gamma
                best_C = C

    return best_gamma, best_C, best_auc


def main():
    # Synthetic concentric-circle data.
    x_data, y_data = datasets.make_circles(
        n_samples=NUM_DATAPOINTS,
        noise=0.15,
        factor=0.6,
    )

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(
        x_data[:, 0],
        x_data[:, 1],
        c=y_data,
        edgecolors="grey",
        cmap="Blues",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Feature 0")
    ax.set_ylabel("Feature 1")
    ax.legend(
        handles=scatter.legend_elements()[0],
        labels=["outside (0)", "inside (1)"],
    )
    plt.show()

    # 80/20 stratified train/test split.
    x_train, x_test, y_train, y_test = model_selection.train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_data,
    )

    linear_clf = linear_model.LogisticRegression()
    linear_clf.fit(x_train, y_train)

    svm_clf = svm.SVC(gamma=2, C=1, probability=True)
    svm_clf.fit(x_train, y_train)

    print(
        "Accuracy — logistic:",
        linear_clf.score(x_test, y_test),
        "| SVM:",
        svm_clf.score(x_test, y_test),
    )

    # Decision-surface comparison.
    fig, ax = plt.subplots(ncols=2, figsize=(10, 7))
    fig.colorbar(
        cm.ScalarMappable(cmap="Blues"),
        ticks=np.linspace(0, 1, 11),
        ax=ax,
    )
    plot_decision_surface(
        linear_clf,
        (-1.5, 1.5),
        (-1.5, 1.5),
        ax[0],
        "Logistic Regression",
        x_data,
        y_data,
    )
    plot_decision_surface(
        svm_clf,
        (-1.5, 1.5),
        (-1.5, 1.5),
        ax[1],
        "SVM",
        x_data,
        y_data,
    )
    plt.show()

    # ROC curves and ROC-AUC.
    linear_preds = linear_clf.predict_proba(x_test)[:, 1]
    svm_preds = svm_clf.predict_proba(x_test)[:, 1]

    linear_curve = metrics.roc_curve(y_test, linear_preds)
    svm_curve = metrics.roc_curve(y_test, svm_preds)

    plt.plot(
        linear_curve[0],
        linear_curve[1],
        label="Logistic ROC-AUC: %.2f"
        % metrics.roc_auc_score(y_test, linear_preds),
    )
    plt.plot(
        svm_curve[0],
        svm_curve[1],
        label="SVM ROC-AUC: %.2f"
        % metrics.roc_auc_score(y_test, svm_preds),
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    # Hyperparameter tuning uses validation data, not test data.
    best_gamma, best_C, best_auc = tune_svm(x_train, y_train)

    print("Best gamma:", best_gamma)
    print("Best C:", best_C)
    print("Validation ROC-AUC:", best_auc)

    # Retrain on the full original training set with selected hyperparameters.
    best_svm = svm.SVC(
        gamma=best_gamma,
        C=best_C,
        probability=True,
    )
    best_svm.fit(x_train, y_train)

    # Final evaluation on the untouched test set.
    test_preds = best_svm.predict_proba(x_test)[:, 1]
    test_auc = metrics.roc_auc_score(y_test, test_preds)
    print("Test ROC-AUC:", test_auc)


if __name__ == "__main__":
    main()
