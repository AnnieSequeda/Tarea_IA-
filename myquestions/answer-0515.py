import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


def pipeline_pca_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha: float = 1.0
) -> dict:

    imputer = SimpleImputer(strategy='mean')

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)

    pca = PCA(n_components=0.95)

    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca = pca.transform(X_test_sc)

    model = Ridge(alpha=alpha)

    model.fit(X_train_pca, y_train)

    predicciones = model.predict(X_test_pca)

    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    r2 = r2_score(y_test, predicciones)

    return {
        "n_componentes": int(pca.n_components_),
        "rmse": float(rmse),
        "r2": float(r2),
        "predicciones": predicciones
    }
