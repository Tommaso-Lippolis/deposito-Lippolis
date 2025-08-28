from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool
from qa_ai_agent.tools.custom_tool import rag_retrieval
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


web_search_tool = SerperDevTool(n_results=3)
rag_tool = rag_retrieval

@CrewBase
class QaCrew():
    """Crew for orchestrated RAG and web search with summarization"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def rag_prompt_rewriter(self) -> Agent:
        return Agent(
            config=self.agents_config['rag_prompt_rewriter'],
            verbose=True
        )

    @agent
    def rag_retriever(self) -> Agent:
        return Agent(
            config=self.agents_config['rag_retriever'],
            verbose=True,
            tools=[rag_tool]
        )

    @agent
    def web_prompt_rewriter(self) -> Agent:
        return Agent(
            config=self.agents_config['web_prompt_rewriter'],
            verbose=True
        )

    @agent
    def web_searcher(self) -> Agent:
        return Agent(
            config=self.agents_config['web_searcher'],
            verbose=True,
            tools=[web_search_tool]
        )

    @agent
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['summarizer'],
            verbose=True
        )

    @task
    def rag_prompt_task(self) -> Task:
        return Task(
            config=self.tasks_config['rag_prompt_task']
        )

    @task
    def rag_retrieval_task(self) -> Task:
        return Task(
            config=self.tasks_config['rag_retrieval_task']
        )

    @task
    def web_prompt_task(self) -> Task:
        return Task(
            config=self.tasks_config['web_prompt_task']
        )

    @task
    def web_search_task(self) -> Task:
        return Task(
            config=self.tasks_config['web_search_task']
        )

    @task
    def summarization_task(self) -> Task:
        return Task(
            config=self.tasks_config['summarization_task'],
            output_file='output/report.md'
        )
    
    @crew
    def rag_crew(self) -> Crew:
        """Creates the orchestrated RAG/WebSearch + summarization crew"""
        return Crew(
            agents=[self.rag_prompt_rewriter(),
                    self.rag_retriever(),
                    self.summarizer()],
            tasks=[self.rag_prompt_task(),
                   self.rag_retrieval_task(),
                   self.summarization_task()],
            process=Process.sequential,
            verbose=True,
        )

    @crew
    def web_search_crew(self) -> Crew:
        """Creates the orchestrated web search + summarization crew"""
        return Crew(
            agents=[self.web_prompt_rewriter(),
                    self.web_searcher(),
                    self.summarizer()],
            tasks=[self.web_prompt_task(),
                   self.web_search_task(),
                   self.summarization_task()],
            process=Process.sequential,
            verbose=True,
        )
