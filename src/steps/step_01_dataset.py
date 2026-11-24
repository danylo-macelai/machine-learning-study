
import pandas as pd
import numpy as np


def create_synthetic_people_dataset() -> pd.DataFrame:

    df = pd.DataFrame({

        "ID": list(range(1, 31)),

        # MINMAXSCALER
        "age": [
            22, 25, 28, 30, 32,
            35, 38, 40, 42, 450,
            23, 27, 31, 34, 37,
            41, 44, 48, 50, 55,
            26, 29, 33, 36, 39,
            43, 22, 32, 25, 65
        ],

        # STANDARDSCALER
        # Contém 1 NULL (ID 26) e 1 outlier (ID 30).
        # Aplicar SimpleImputer antes do StandardScaler.
        # O NULL será imputado pela mediana ou média.
        # O outlier permite avaliar o impacto de valores extremos.
        "income": [
            1800, 2200, 2800, 3200, 3500,
            4000, 4500, 5000, 5500, 6000,
            2000, 2600, 3100, 3700, 4200,
            4800, 5300, 5800, 6500, 7200,
            2400, 2900, 3400, 3900, 4400,
            np.nan, 1800, 3500, 2200, 80000
        ],

        # ONEHOTENCODER
        "job": [
            "CLT", "Autonomo", "Servidor", "CLT", "Empresario",
            "Autonomo", "Servidor", "CLT", "Empresario", "CLT",
            "CLT", "Autonomo", "Servidor", "CLT", "Empresario",
            "CLT", "Autonomo", "Servidor", "CLT", "Empresario",
            "CLT", "Autonomo", "Servidor", "CLT", "Empresario",
            "CLT", "CLT", "Empresario", "Autonomo", "Servidor"
        ],

        # LABELENCODER
        # Codificação de uma segunda variável categórica.
        "marital": [
            "Solteiro", "Casado", "Solteiro", "Casado", "Divorciado",
            "Casado", "Solteiro", "Viuvo", "Casado", "Solteiro",
            "Solteiro", "Casado", "Divorciado", "Solteiro", "Casado",
            "Viuvo", "Solteiro", "Casado", "Divorciado", "Casado",
            "Solteiro", "Casado", "Solteiro", "Viuvo", "Casado",
            "Divorciado", "Solteiro", "Divorciado", "Casado", "Viuvo"
        ],

        # ORDINALENCODER
        # Escolaridade possui uma ordem hierárquica.
        "education": [
            "Medio", "Superior", "Medio", "Superior", "Superior",
            "Pos-graduacao", "Superior", "Medio", "Superior", "Pos-graduacao",
            "Medio", "Superior", "Medio", "Superior", "Pos-graduacao",
            "Superior", "Medio", "Superior", "Pos-graduacao", "Superior",
            "Medio", "Superior", "Superior", "Medio", "Pos-graduacao",
            "Superior", "Medio", "Superior", "Superior", "Pos-graduacao"
        ],

        # Variável numérica para complementar a previsão de gastos.
        "dependents": [
            0, 0, 1, 1, 2,
            2, 2, 3, 2, 3,
            0, 1, 1, 2, 2,
            3, 2, 3, 3, 40,
            0, 1, 2, 1, 2,
            2, 0, 2, 0, 4
        ],

        # Variável categórica nominal.
        "region": [
            "Sudeste", "Sul", "Nordeste", "Sudeste", "Centro-Oeste",
            "Sul", "Sudeste", "Nordeste", "Sul", "Sudeste",
            "Norte", "Nordeste", "Sudeste", "Sul", "Centro-Oeste",
            "Sudeste", "Norte", "Nordeste", "Sul", "Sudeste",
            "Nordeste", "Centro-Oeste", "Sudeste", "Sul", "Norte",
            "Sudeste", "Sudeste", "Centro-Oeste", "Sul", "Sudeste"
        ],

        # TARGET DA REGRESSÃO
        "spending": [
            1200, 1500, 1800, 2100, 2300,
            2600, 2800, 3000, 3300, 3500,
            1400, 1700, 2000, 2200, 2500,
            2700, 2900, 3200, 3500, 3800,
            1600, 1900, 2100, 2400, 2600,
            2800, 1200, 2300, 1500, 30000
        ]

    })

    return df


# ==========================================
# DUPLICADOS INTENCIONAIS
# ==========================================
# IDs 27, 28 e 29 devem repetir os registros
# 1, 5 e 2, respectivamente.


# ==========================================
# 101 - PESSOA ECONÔMICA
# ==========================================

def create_people_economic() -> pd.DataFrame:

    return pd.DataFrame({
        "ID": [101],
        "age": [30],
        "income": [5000],
        "dependents": [0],
        "job": ["CLT"],
        "marital": ["Solteiro"],
        "education": ["Superior"],
        "region": ["Sudeste"]
    })


# ==========================================
# 102 - PESSOA CONSUMISTA
# ==========================================

def create_people_high_spending() -> pd.DataFrame:

    return pd.DataFrame({
        "ID": [102],
        "age": [35],
        "income": [12000],
        "dependents": [2],
        "job": ["Empresario"],
        "marital": ["Casado"],
        "education": ["Superior"],
        "region": ["Sudeste"]
    })


# ==========================================
# 103 - RENDA ALTA E GASTOS REDUZIDOS
# ==========================================

def create_people_high_income_low_spending() -> pd.DataFrame:

    return pd.DataFrame({
        "ID": [103],
        "age": [45],
        "income": [20000],
        "dependents": [0],
        "job": ["Servidor"],
        "marital": ["Viuvo"],
        "education": ["Pos-graduacao"],
        "region": ["Sul"]
    })


# ==========================================
# 104 - RENDA BAIXA E GASTOS ELEVADOS
# ==========================================

def create_people_low_income_high_spending() -> pd.DataFrame:

    return pd.DataFrame({
        "ID": [104],
        "age": [32],
        "income": [2500],
        "dependents": [3],
        "job": ["Autonomo"],
        "marital": ["Casado"],
        "education": ["Medio"],
        "region": ["Nordeste"]
    })


# ==========================================
# 105 - PESSOA COM FAMÍLIA
# ==========================================

def create_people_large_family() -> pd.DataFrame:

    return pd.DataFrame({
        "ID": [105],
        "age": [38],
        "income": [6500],
        "dependents": [4],
        "job": ["CLT"],
        "marital": ["Casado"],
        "education": ["Superior"],
        "region": ["Centro-Oeste"]
    })


# ==========================================
# 106 - JOVEM COM RENDA ELEVADA
# ==========================================

def create_people_young_high_income() -> pd.DataFrame:

    return pd.DataFrame({
        "ID": [106],
        "age": [24],
        "income": [10000],
        "dependents": [0],
        "job": ["Empresario"],
        "marital": ["Solteiro"],
        "education": ["Superior"],
        "region": ["Sudeste"]
    })


# ==========================================
# 107 - PESSOA IDOSA
# ==========================================

def create_people_elderly() -> pd.DataFrame:

    return pd.DataFrame({
        "ID": [107],
        "age": [75],
        "income": [7000],
        "dependents": [0],
        "job": ["Servidor"],
        "marital": ["Viuvo"],
        "education": ["Medio"],
        "region": ["Sul"]
    })


# ==========================================
# 108 - RENDA MUITO ELEVADA
# ==========================================

def create_people_extremely_high_income() -> pd.DataFrame:

    return pd.DataFrame({
        "ID": [108],
        "age": [42],
        "income": [150000],
        "dependents": [2],
        "job": ["Empresario"],
        "marital": ["Casado"],
        "education": ["Pos-graduacao"],
        "region": ["Sudeste"]
    })


# ==========================================
# 109 - PERFIL DE RENDA BAIXA
# ==========================================

def create_people_very_low_income() -> pd.DataFrame:

    return pd.DataFrame({
        "ID": [109],
        "age": [23],
        "income": [1000],
        "dependents": [1],
        "job": ["CLT"],
        "marital": ["Solteiro"],
        "education": ["Medio"],
        "region": ["Norte"]
    })


# ==========================================
# 110 - PERFIL INTERMEDIÁRIO
# ==========================================

def create_people_average_profile() -> pd.DataFrame:

    return pd.DataFrame({
        "ID": [110],
        "age": [40],
        "income": [4500],
        "dependents": [2],
        "job": ["Autonomo"],
        "marital": ["Casado"],
        "education": ["Superior"],
        "region": ["Sul"]
    })


# ==========================================
# AGRUPAMENTO DOS TESTES
# ==========================================

prediction_tests = {
    "economic": create_people_economic(),
    "high_spending": create_people_high_spending(),
    "high_income_low_spending": create_people_high_income_low_spending(),
    "low_income_high_spending": create_people_low_income_high_spending(),
    "large_family": create_people_large_family(),
    "young_high_income": create_people_young_high_income(),
    "elderly": create_people_elderly(),
    "extremely_high_income": create_people_extremely_high_income(),
    "very_low_income": create_people_very_low_income(),
    "average_profile": create_people_average_profile()
}
