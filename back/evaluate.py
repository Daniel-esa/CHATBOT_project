import pandas as pd
import mlflow
from mlflow.metrics.genai import EvaluationExample, faithfulness
import os
from dotenv import load_dotenv
from mlflow.metrics.genai import EvaluationExample, relevance

#from func import get_rag_chain

load_dotenv()  # Charge la clé depuis .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ----------------------------------------------------------------------------
# 3. Constantes modèles
# ----------------------------------------------------------------------------


# === Now you can use it exactly like OpenAI SDK ===
AZURE_AOAI_MODEL_GPT3_TURBO = "gpt35turbo"
AZURE_AOAI_MODEL_GPT4O = "gpt-4o"
AZURE_AOAI_API_VERSION = "2024-08-01-preview"
AZURE_AOAI_MODEL_GPT4OMINI = "gpt-4o-mini"
AZURE_EMBEDDING_MODEL = "text-embedding-ada-002"




# ----------------------------------------------------------------------------
# 5. Faithfulness Evaluation    
# ----------------------------------------------------------------------------
# This example demonstrates how to use the faithfulness metric to evaluate a model's responses

faithfulness_examples = [
  EvaluationExample(
      input="How much is the spend in Algeria in 2023? ",
      output="Spend in Algeria is 1M€ in 2023.",
      score=5,
      justification="The output provides a working solution, amount and year that is provided in the context.",
      grading_context = {   
          "context": "Algeria is one of the countries of the group, we noticed a new spend in 2023 of 10M€. with a 10% increase compared to 2022. The total spend in the group is 10M€ in 2023, with a 5% increase compared to 2022."
      },
  ),

  EvaluationExample(
      input="give th eprocurement KPIs of the group",
      output="The procurement KPIs of the group are: Spend in Algeria is 1M€ in 2023, Spend in France is 2M€ in 2023, Spend in Germany is 3M€ in 2023, Spend in Italy is 4M€ in 2023.",
      score=5,
      justification="The output provides a working solution that is using the context provided.",
      grading_context={
          "context": "Algeria is one of the countries of the group, we noticed a new spend in 2023 of 10M€. with a 10% increase compared to 2022. The total spend in the group is 10M€ in 2023, with a 5% increase compared to 2022. The procurement KPIs of the group are: Spend in Algeria is 1M€ in 2023, Spend in France is 2M€ in 2023, Spend in Germany is 3M€ in 2023, Spend in Italy is 4M€ in 2023."
      },
  ),
]

faithfulness_metric = faithfulness(model=AZURE_AOAI_MODEL_GPT4O, examples=faithfulness_examples)
relevance_metric = relevance(model=AZURE_AOAI_MODEL_GPT4O)
#print(relevance_metric)
#print(faithfulness_metric)

def run_eval(rag_chain):  

# ----------------------------------------------------------------------------
# 4. DataFrame d'évaluation
# ----------------------------------------------------------------------------
# Create a DataFrame for evaluation questions
    eval_df = pd.DataFrame(
    {
        "questions": [
            "how much spend in france?",
            "provides me with spend in algeria?",
            "give me procurement kpis of italy?",
        ],
    }
    )



# ----------------------------------------------------------------------------
# 4. Fonction de modèle
# ----------------------------------------------------------------------------
    def model(input_df: pd.DataFrame) -> list[str]:
        """Pour chaque question, on fait la retrieval + answer via RAG."""
        outputs = []
        for _, row in input_df.iterrows():
            q = row["questions"]
            # la chaîne retourne un dict { 'result': ..., 'source_documents': [...] }
            result = rag_chain.invoke({"query": q})
            outputs.append(result)
        return outputs


# ----------------------------------------------------------------------------
# 6. Model Evaluation
# ----------------------------------------------------------------------------

    results = mlflow.evaluate(
    model,
    eval_df,
    model_type="question-answering",
    evaluators="default",
    predictions="result",
    extra_metrics=[faithfulness_metric, relevance_metric, mlflow.metrics.latency()],
    evaluator_config={
        "col_mapping": {
            "inputs": "questions",
            "context": "source_documents",
        }
    },
    )
    print(results.metrics)