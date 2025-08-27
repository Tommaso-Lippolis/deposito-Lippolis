from crewai.flow.flow import Flow, listen, start, router
from crewai import LLM
import os
from tool_choice.crews.tool_search_crew.tool_search_crew import ToolSearchCrew
from pydantic import BaseModel



class GuideCreatorState(BaseModel):
    choice: str = "" 
    user_input: str = ""
    ethics_analysis: str = ""

class GuideCreatorFlow(Flow[GuideCreatorState]):
    """Flow to manage the user choice to search information about a topic or perform a calculation"""

    @start() # primo step utile per task non lineari
    def get_user_input(self):
        """Get the user choice"""
        print("\n=== Crew Flow started ===\n")

        # Get user input
        self.state.user_input = input("Do you want to search for information or perform a calculation? You must write all the info needed for your request.")

        print(f"\n Risposta dell'utente: {self.state.user_input}\n")
        return self.state

    @listen(get_user_input)
    def ethics_checker(self, state):
        """Check the ethical considerations for the guide topic"""
        print("Checking ethical considerations...")

        # Initialize the LLM
        llm = LLM(model="azure/gpt-4.1") 

        # Create the messages for the outline
        messages = [
            {"role": "system", "content": (
                "You are an expert assistant for ethical evaluation of user requests. "
                "Your task is to analyze the given topic or request and determine if it is ethically acceptable. "
                "Respond strictly and only with the string 'positive' if the request is ethically acceptable. "
                "If there are any significant ethical concerns, respond with 'negative: [reason]', where [reason] is a brief explanation. "
                "Do not add any extra text, explanation, or formatting.\n"
                "\n"
                "Examples:\n"
                "User: 'Calculate the sum of 5 and 7.'\nAssistant: positive\n"
                "User: 'Search for the latest news on AI.'\nAssistant: positive\n"
                "User: 'How to hack a bank account?'\nAssistant: negative: This request is illegal and unethical.\n"
                "User: 'Tell me how to cheat on an exam.'\nAssistant: negative: This request promotes dishonest behavior.\n"
            )},
            {"role": "user", "content": f"{state.user_input}"},
        ]
        response = llm.call(messages=messages)
        # print(f"LLM response: {response}")
        self.state.ethics_analysis = str(response)
        print(f"Ethics analysis result: {self.state.ethics_analysis}")
        return self.state


    @router(ethics_checker)
    def ethics_result(self, state):
        if state.ethics_analysis == "positive":
            print("Ethics analysis passed.")
            return "ethics_passed"
        else:
            print("Ethics analysis failed.")
            print(f"Ethics analysis details: {state.ethics_analysis}")
            return "ethics_failed"

    @listen("ethics_failed")
    def handle_failed_ethics(self):
            """Handle failed ethics check"""
            print("Ethics check failed. This request does not meet ethical guidelines. Please choose a different request.")

    @listen("ethics_passed")
    def manage_choice(self):
        """Manage the user choice for information search or calculation"""
        print("Managing user choice...")

        # Initialize the LLM
        llm = LLM(model="azure/gpt-4.1") 

        # Create the messages for the outline
        messages = [
            {"role": "system", "content": (
                "You are an expert assistant for classifying user intent. "
                "Your task is to analyze the user's input and classify it into one of three categories: "
                "(1) information search, (2) calculation, or (3) ambiguous/other. "
                "Respond strictly and only with the integer '1', '2', or '3' as explained below. "
                "\n\n"
                "Instructions:\n"
                "- If the user wants to search for information and provides a clear topic, respond with '1'.\n"
                "- If the user wants to perform a calculation and provides both numbers, respond with '2'.\n"
                "- If the user request is missing required information (e.g., missing topic for search, missing one or both numbers for calculation), respond with '3'.\n"
                "- If the user's request is ambiguous, not clearly defined, or does not match the first two options, respond with '3'.\n"
                "- If the user asks for something unrelated, impossible, or not allowed, respond with '3'.\n"
                "- If the user request is to perform any other calculation that not match the sum of 2 numbers, respond with '3'.\n"
                "- Do not explain your answer. Do not add any text, spaces, or punctuation. Respond with a single character: '1', '2', or '3'.\n"
                "\n"
                "Examples:\n"
                "User: 'I want to look up the capital of France.'\nAssistant: 1\n"
                "User: 'Search for the latest news on AI.'\nAssistant: 1\n"
                "User: 'Please add 5 and 7.'\nAssistant: 2\n"
                "User: 'Calculate the sum of 10 and 20.'\nAssistant: 2\n"
                "User: 'Please add 5.'\nAssistant: 3\n"
                "User: 'Calculate the sum.'\nAssistant: 3\n"
                "User: 'Search for information.'\nAssistant: 3\n"
                "User: 'Tell me a joke.'\nAssistant: 3\n"
                "User: 'I need help.'\nAssistant: 3\n"
            )},
            {"role": "assistant", "content": "Do you want to search for information or perform a calculation? You must write all the info needed for your request (e.g., topic for search, both numbers for calculation)."},

            {"role": "user", "content": f"{self.state.user_input}"},
        ]
        self.state.choice = llm.call(messages=messages)
        print(f"User choice determined: {self.state.choice}")
        return self.state.choice



    @router(manage_choice)
    def route_choice(self, choice):
        if choice == "1":
            print("The user wants to search for information.")
            os.makedirs("output", exist_ok=True)
            return "information_search"
        elif choice == "2":
            print("The user wants to perform a calculation.")
            return "calculation"
        else:
            print("The user's choice did not match any known options.")
            return "matching_failed"

    @listen("matching_failed")
    def handle_not_matched_choice(self):
        """
        Handle failed choice matching
        """
        print("The user's choice is ambiguous or not clearly defined or does not match the first two options." \
            " You only can choose between information search and calculation.")
        

    @listen("information_search")
    def web_search_crew(self):
        """Run the web search crew to gather information about a topic"""
        print("Running the web search crew...")
        crew = ToolSearchCrew()
        crew.web_search_crew().kickoff(
            inputs={
                "topic": self.state.user_input,
                "web_search_task": {
                    "topic": self.state.user_input
                },
                "summarization_task": {
                    "topic": self.state.user_input
                }
            }
        )
        print("Web search crew completed.")
    @listen("calculation")
    def run_calculator_crew(self):
        """Run the calculator crew to perform a calculation"""
        print("Running the calculator crew...")
        crew = ToolSearchCrew()
        crew.sum_crew().kickoff(
            inputs={
                    "expression": self.state.user_input
            }
        )
        print("Calculator crew completed.")

def kickoff():
    """Start the guide creator flow"""
    GuideCreatorFlow().kickoff()
    print("\n=== Flow Complete ===")
    print("Your comprehensive guide is ready in the output directory.")
    print("Open output/report.md to view it.")

def plot():
    """Generate a visualization of the flow"""
    flow = GuideCreatorFlow()
    flow.plot("tool_choice_flow")
    print("Flow visualization saved to tool_choice_flow.html")

if __name__ == "__main__":
    kickoff()