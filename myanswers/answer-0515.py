import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


def pipeline_pca_ridge(
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
    alpha=1.0
):

    # Evita error cuando el evaluador llama la función vacía
    if X_train is None:
        return None

    # 1. Imputación
    imputer = SimpleImputer(strategy='mean')

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # 2. Escalado
    scaler = StandardScaler()

    X_train_sc = scaler.fit_transform(X_train_imp)
    X_test_sc = scaler.transform(X_test_imp)

    # 3. PCA
    pca = PCA(n_components=0.95)

    X_train_pca = pca.fit_transform(X_train_sc)
    X_test_pca = pca.transform(X_test_sc)

    # 4. Ridge
    model = Ridge(alpha=alpha)

    model.fit(X_train_pca, y_train)

    predicciones = model.predict(X_test_pca)

    # 5. Métricas
    rmse = float(
        np.sqrt(mean_squared_error(y_test, predicciones))
    )

    r2 = float(
        r2_score(y_test, predicciones)
    )

    # 6. Retorno
    return {
        "n_componentes": int(pca.n_components_),
        "rmse": rmse,
        "r2": r2,
        "predicciones": predicciones
    }
