"""LLM-based text to exogenous array transformer for sktime."""

import json

import pandas as pd
from groq import Groq
from sktime.transformations.base import BaseTransformer


class LLMTextEventFeaturizer(BaseTransformer):
    """
    Transforms unstructured textual event logs into multivariate numerical features 
    using a Large Language Model (Foundation Model).
    
    This acts as a native sktime transformer for data preparation, mapping text 
    logs directly into explicit exogenous signals (X) for downstream predictors.
    """
    
    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "univariate-only": False,
        "fit_is_empty": True,
    }

    def __init__(
        self,
        api_key: str = None,
        text_column: str = None,
        feature_schema: dict = None,
        model: str = "llama-3.3-70b-versatile",
        backend: any = None,
        batch_size: int = 10,
    ):
        """
        Parameters
        ----------
        api_key : str, optional
            Groq API Key.
        text_column : str
            The name of the column in X containing the raw unstructured text.
        feature_schema : dict
            A dictionary defining the features to extract.
        model : str, optional
            Groq Supported Model.
        backend : any, optional
            LangChain ChatModel object.
        batch_size : int
            Number of text events to process in a single LLM call.
        """
        self.api_key = api_key
        self.text_column = text_column
        self.feature_schema = feature_schema
        self.model = model
        self.backend = backend
        self.batch_size = batch_size
        super().__init__()

    def _call_llm(self, sys_prompt: str, user_prompt: str, temperature: float = 0):
        """Internal bridge to call either Groq or LangChain backend."""
        if self.backend:
            from langchain_core.messages import SystemMessage, HumanMessage
            response = self.backend.invoke([
                SystemMessage(content=sys_prompt),
                HumanMessage(content=user_prompt)
            ])
            return response.content.strip()
        else:
            client = Groq(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()

    def _transform(self, X, y=None):
        if self.text_column not in X.columns:
            raise ValueError(f"Column '{self.text_column}' not found in X.")
            
        client = Groq(api_key=self.api_key)
        
        schema_json_str = json.dumps(self.feature_schema, indent=2)
        system_prompt = f"""You are an expert data science assistant analyzing textual time-series events.
Your task is to extract specified numerical and categorical features from a list of unstructured text logs.

Schema to extract for EACH event:
{schema_json_str}

Respond with ONLY a valid JSON LIST of objects. Each object must map the exact keys in the schema to numeric values.
The list MUST match the length and order of the input events provided.
If a feature cannot be determined for a specific event, return 0.0 or a reasonable neutral metric for that object.
Do NOT include markdown formatting or additional reasoning text."""

        results = [None] * len(X)
        indices = list(range(len(X)))
        
        # Process in batches
        for i in range(0, len(X), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_texts = [str(X.iloc[idx][self.text_column]) for idx in batch_indices]
            
            # Filter out empty/NaN for the batch request
            # But we must maintain order in the response, so we still send dummy placeholders if needed?
            # Actually, it's safer to send everything and let the LLM handle it, or handle it ourselves.
            
            user_payload = "\n".join([f"Event {j}: '{t}'" for j, t in enumerate(batch_texts)])
            
            try:
                raw_out = self._call_llm(
                    system_prompt, 
                    f"Extract features for these {len(batch_texts)} events:\n{user_payload}",
                    temperature=0.0
                )
                raw_out = raw_out.replace("```json", "").replace("```", "").strip()
                
                # Basic JSON list attempt
                try:
                    parsed_list = json.loads(raw_out)
                    if not isinstance(parsed_list, list):
                        parsed_list = [parsed_list] # Fallback for single item
                except Exception:
                    # If the whole batch fails, fallback to neutral for each
                    print(f"Warning: Batch parsing failed for indices {batch_indices}. Falling back to default.")
                    parsed_list = []

                # Map back to results
                for j, idx in enumerate(batch_indices):
                    if j < len(parsed_list):
                        parsed = parsed_list[j]
                    else:
                        parsed = {}
                    
                    filled_parsed = {}
                    for k in self.feature_schema.keys():
                        val = parsed.get(k, 0.0)
                        try:
                            filled_parsed[k] = float(val)
                        except (ValueError, TypeError):
                            filled_parsed[k] = 0.0
                    results[idx] = filled_parsed
                    
            except Exception as e:
                print(f"Warning: API error for batch starting at index {i}. Error: {e}")
                for idx in batch_indices:
                    results[idx] = {k: 0.0 for k in self.feature_schema.keys()}

        X_transformed = X.copy()
        features_df = pd.DataFrame(results, index=X.index)
        
        # Drop the original unstructured text column
        X_transformed = X_transformed.drop(columns=[self.text_column])
        
        # Bind the newly extracted structural features
        X_transformed = pd.concat([X_transformed, features_df], axis=1)
        
        # Cast to numeric to ensure seamless sktime estimator forwarding
        return X_transformed.apply(pd.to_numeric, errors='coerce').fillna(0.0)
