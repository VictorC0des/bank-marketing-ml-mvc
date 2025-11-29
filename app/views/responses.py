from pydantic import BaseModel, Field
from typing import Literal
from enum import Enum

# ===== Tipos de Modelos =====
class ModelType(str, Enum):
    DECISION_TREE = "decision_tree"
    DEEP_LEARNING = "deep_learning"

# ===== Tipos de Entrada =====
Job = Literal["admin.","blue-collar","entrepreneur","housemaid","management",
              "retired","self-employed","services","student","technician",
              "unemployed","unknown"]
Marital = Literal["married","single","divorced","unknown"]
Education = Literal["unknown","primary","secondary","tertiary"]
YNUNK = Literal["yes","no","unknown"]
Contact = Literal["cellular","telephone","unknown"]
Month = Literal["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
Poutcome = Literal["success","failure","other","unknown"]

class InputData(BaseModel):
    age: int
    job: Job
    marital: Marital
    education: Education
    default: YNUNK
    balance: float
    housing: YNUNK
    loan: YNUNK
    contact: Contact
    day: int = Field(ge=1, le=31)
    month: Month
    duration: int = Field(ge=0)
    campaign: int = Field(ge=1)
    pdays: int
    previous: int = Field(ge=0)
    poutcome: Poutcome
