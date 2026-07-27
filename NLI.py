#Using a NLI model (MoitzLaurer)

from pydantic.v1 import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class NLIEvaluation(BaseModel):
    model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    premise: str = "As a best practice, it is recommended not to share social security number. But in some scenarios social security number" \
              "can be shared "
    hypothesis: str = "Yes, we can share social security number."

    def get_nli_score(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        input = tokenizer(self.premise, self.hypothesis, truncation=True, return_tensors="pt")
        output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
        prediction = torch.softmax(output["logits"][0], -1).tolist()
        label_names = ["entailment", "neutral", "contradiction"]
        prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
        return prediction

if __name__=="__main__":
    eval = NLIEvaluation()
    prediction = eval.get_nli_score()
    print(prediction)