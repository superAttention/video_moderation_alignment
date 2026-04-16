import re
from rewards.base import RewardFn

def extract_answer(text: str) -> str:
    match = re.search("\s*(-?[\d,]+)", text) 
    return match.group(1).replace(",", "") if match else ""

class MathReward:                                                              

      def __init__(self, references: list[str]):                                 
          self.references = references                                           
                                              
      def __call__(self, prompts: list[str], completions: list[str]) -> list[float]:                                                                   
          return [                                                               
              1.0 if extract_answer(completion) == ref else 0.0
              for completion, ref in zip(completions, self.references)           
          ]                                                                      
                                            

