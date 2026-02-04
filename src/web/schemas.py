from pydantic import BaseModel

class UserInputRequest(BaseModel):
    study: float
    social: float
    netflix: float
    attendance: float
    sleep: float
    exercise: float
    mental: float