import streamlit as st
import pandas as pd
import unicodedata
from pycaret.classification import *
import pickle

# Função para remover acentos e substituir espaços por underlines
def normalize_text(text):
    if isinstance(text, str):
        text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ASCII")
        return text.replace(" ", "_")
    return text

# Configuração da página no Streamlit
st.title("📊 Treinamento de Modelo de Crédito")

# Upload do arquivo .ftr
uploaded_file = st.file_uploader("Escolha um arquivo .ftr", type=["ftr"])

if uploaded_file is not None:
    # Carregar o arquivo
    df = pd.read_feather(uploaded_file)
    
    # Remover colunas irrelevantes
    df.drop(columns=['data_ref', 'index'], inplace=True, errors='ignore')
    
    # Normalizar colunas
    df.columns = [normalize_text(col) for col in df.columns]
    
    # Normalizar valores categóricos
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].apply(normalize_text)
    
    st.write("📋 Dados carregados com sucesso:")
    st.dataframe(df.head())
    
    # Configurar PyCaret
    clf = setup(
        data=df,
        target='mau',
        train_size=0.8,
        session_id=42,
        normalize=True,
        feature_selection=True,
        remove_multicollinearity=True,
        transformation=True,
        fix_imbalance=True
    )
    
    # Treinar o modelo
    st.write("🔄 Treinando o modelo...")
    modelo = create_model('lightgbm')
    modelo_tuned = tune_model(modelo)
    
    # Fazer previsões
    predictions = predict_model(modelo_tuned)
    
    st.write("✅ Treinamento concluído! Veja as previsões abaixo:")
    st.dataframe(predictions)
    
    # Salvar o modelo treinado
    save_model(modelo_tuned, 'modelo_credit_scoring_lgbm')
    
    # Salvar previsões em arquivo .pkl
    with open("model_final.pkl", "wb") as f:
        pickle.dump(predictions, f)
    
    st.success("✅ Previsões salvas em 'model_final.pkl'")
    st.download_button("Baixar arquivo de previsões", data=open("model_final.pkl", "rb"), file_name="model_final.pkl")
