#Using a NLI model (MoitzLaurer)

from pydantic.v1 import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")