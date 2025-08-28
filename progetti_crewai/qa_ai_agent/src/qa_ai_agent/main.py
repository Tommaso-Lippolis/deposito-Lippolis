from crewai.flow.flow import Flow, listen, start, router, or_, and_
from crewai import LLM
import os
from qa_ai_agent.crews.qa_crew.qa_crew import QaCrew
from pydantic import BaseModel



class GuideCreatorState(BaseModel):
    topic: str = "" 
    ethics_analysis: str = ""
    choice: str = ""

class GuideCreatorFlow(Flow[GuideCreatorState]):
    """Flow to manage the user choice to search information about a topic or perform a calculation"""

    @start(or_("retry", "retry_from_start"))
    def get_user_input(self):
        """Get the user choice"""
        print("\n=== Crew Flow started ===\n")

        # Get user input
        self.state.topic = input("What topic would you like to search? ")

        print(f"\n Risposta dell'utente: {self.state.topic}\n")
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
            {"role": "user", "content": f"{state.topic}"},
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

    @router("ethics_failed")
    def handle_failed_ethics(self):
        """Handle failed ethics check"""
        print("Ethics check failed. This request does not meet ethical guidelines. Please choose a different request.")
        return "retry"

    @listen("ethics_passed")
    def manage_choice(self):
        """Manage the user choice for information search or calculation"""
        print("Managing user choice...")

        # Initialize the LLM
        llm = LLM(model="azure/gpt-4.1") 

        # Create the messages for the outline
        messages = [
            {"role": "system", "content": (
                "You are an expert intent classifier for information search. "
                "Your task is to analyze the user's input and classify it into one of three categories: "
                "(1) information search strictly and unambiguously about smartphones, "
                "(2) information search about any other topic not related to smartphones, "
                "(3) any request that is not an information search (e.g., calculation, joke, help, ambiguous, missing topic, illegal, unethical, or impossible). "
                "Respond strictly and only with the integer '1', '2', or '3' as explained below. "
                "\n\n"
                "Instructions:\n"
                "- Respond '1' ONLY if the user's request is clearly, directly, and unambiguously about smartphones, smartphone technology, smartphone brands, smartphone features, smartphone usage, smartphone troubleshooting, smartphone comparisons, smartphone news, or any topic strictly related to smartphones.\n"
                "- Respond '2' ONLY if the user's request is a clear information search about a topic that is NOT related to smartphones (e.g., history, geography, science, other electronics, etc.).\n"
                "- Respond '3' if the request is ambiguous, generic, missing a clear topic, not an information search, about calculations, jokes, stories, personal help, hacking, illegal, unethical, impossible actions, or anything not covered by '1' or '2'.\n"
                "- Do not explain your answer. Do not add any text, spaces, or punctuation. Respond with a single character: '1', '2', or '3'.\n"
                "\n"
                "Examples:\n"
                "User: 'What is the best smartphone in 2025?'\nAssistant: 1\n"
                "User: 'How do I update my iPhone?'\nAssistant: 1\n"
                "User: 'Compare Samsung Galaxy and iPhone.'\nAssistant: 1\n"
                "User: 'What is the capital of France?'\nAssistant: 2\n"
                "User: 'Tell me about the history of Rome.'\nAssistant: 2\n"
                "User: 'What is the best smartwatch?'\nAssistant: 2\n"
                "User: 'Help me.'\nAssistant: 3\n"
                "User: 'Search for information.'\nAssistant: 3\n"
                "User: 'Please add 5 and 7.'\nAssistant: 3\n"
                "User: 'Calculate the sum of 10 and 20.'\nAssistant: 3\n"
                "User: 'Tell me a joke.'\nAssistant: 3\n"
                "User: 'How to hack a smartphone?'\nAssistant: 3\n"
                "User: 'Tell me about the latest technology.'\nAssistant: 2\n"
                "User: 'What is a mobile device?'\nAssistant: 2\n"
                "User: 'What is the best phone?'\nAssistant: 2\n"
            )},
            {"role": "assistant", "content": "What topic would you like to search? "},

            {"role": "user", "content": f"{self.state.topic}"},
        ]
        self.state.choice = llm.call(messages=messages)
        print(f"User choice determined: {self.state.choice}")
        return self.state.choice



    @router(manage_choice)
    def route_choice(self, choice):
        if choice == "1":
            print("The user wants to search for information about smartphones.")
            os.makedirs("output", exist_ok=True)
            return "smartphone_rag"
        elif choice == "2":
            print("The user wants to search for information about a topic not related to smartphones.")
            return "general_search"
        else:
            print("The user's choice did not match any known options. Retrying...")
            return "retry_from_start"


    @router("smartphone_rag")
    def run_smartphone_rag(self):
        """
        Run the smartphone RAG crew to gather information about smartphones
        """
        print("Running the smartphone RAG crew...")
        crew = QaCrew()
        response = crew.rag_crew().kickoff(
            inputs={
                "topic": self.state.topic,
                "rag_prompt_task": {
                    "topic": self.state.topic
                },
                "rag_retrieval_task": {
                    "topic": self.state.topic
                },
                "summarization_task": {
                    "topic": self.state.topic
                }
                
            }
        )
        if response.raw == "Non è presente nel contesto fornito.":
            print("RAG crew did not find relevant context.")
            print("Trying to pass the request to the general search crew...")
            return "passing_to_general_search"
        else:
            print("RAG crew completed.")
            return "completed"


    @listen(or_("passing_to_general_search", "general_search"))
    def run_general_search(self):
        """Run the general search crew to perform a search"""
        print("Running the general search crew...")
        crew = QaCrew()
        crew.web_search_crew().kickoff(
            inputs={
                "topic": self.state.topic,
                "web_prompt_task": {
                    "topic": self.state.topic
                },
                "web_search_task": {
                    "topic": self.state.topic
                },
                "summarization_task": {
                    "topic": self.state.topic
                }
            }
        )
        print("General search crew completed.")

def kickoff():
    """Start the guide creator flow"""
    GuideCreatorFlow().kickoff()
    print("\n=== Flow Complete ===")
    print("Your comprehensive guide is ready in the output directory.")
    print("Open output/report.md to view it.")

def plot():
    """Generate a visualization of the flow"""
    flow = GuideCreatorFlow()
    flow.plot("qa_flow")
    print("Flow visualization saved to qa_flow.html")

if __name__ == "__main__":
    kickoff()