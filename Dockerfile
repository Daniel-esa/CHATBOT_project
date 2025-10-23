# 1. Image de base minimaliste avec conda
FROM continuumio/miniconda3

# 2. Copie le YAML et crée l'env 'genai' d'après ce fichier
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && \
    conda clean --all --yes

# 3. Assure-toi que conda est dans le PATH (pour 'conda run')
SHELL ["bash", "-lc"]

# 4. Dossier de travail
WORKDIR /app

# 5. Copie le code de l’application
COPY . /app

# 6. Expose éventuellement le port (pour Streamlit)
EXPOSE 8501

# 7. Commande par défaut : utilise 'conda run' pour exécuter l’app avec l’env genai
CMD ["bash", "-lc", "conda run -n genai python app.py"]
