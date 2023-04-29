from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa.core.nlg.response import TemplatedNaturalLanguageGenerator
from transformers import pipeline
import pandas as pd


# responses = {
#     "utter_greet": [{"text": "Hello, how can I help you?"}, 
#                     {"text": "Hi, hope you are doing well."},
#                     {"text": "Hi, could you tell me your name?"}],
#     "utter_goodbye": [{"text": "Goodbye!"}]
# }
# nlg = TemplatedNaturalLanguageGenerator(responses)
# class ActionHelloWorld(Action):
#     def name(self) -> Text:
#         return "action_hello_world"

#     async def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         print("action_hello_world")
#         message = await nlg.generate("utter_greet", tracker, dispatcher)
#         print(message.get('text'))
#         dispatcher.utter_message(str(message.get('text')))
#         return []

class ActionHelloWorld(Action):
    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("action_hello_world")
        dispatcher.utter_message(str("Hello"))
        return []



df_question_answer_model = tqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq")
travel_df = pd.read_excel("csv/Goa_byNeha_v1.xlsx", sheet_name="How to reach")
restaurent_df = pd.read_csv("csv/restaurent.csv")

class ActionHelloWorld(Action):
    def name(self) -> Text:
        return "action_travel_distance_time"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("action_travel_distance_time")

        question = tracker.latest_message["text"]
        answer = tqa(table=travel_df, query=question)['cells'][0]
        dispatcher.utter_message(answer)
        return []



class ActionHelloWorld(Action):
    def name(self) -> Text:
        return "action_restaurent"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print("action_restaurent")

        question = tracker.latest_message["text"]
        answer = tqa(table=restaurent_df, query=question)['cells'][0]
        dispatcher.utter_message(answer)
        return []
