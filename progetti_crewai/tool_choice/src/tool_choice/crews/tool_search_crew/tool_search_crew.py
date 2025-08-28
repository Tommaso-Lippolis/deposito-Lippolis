# src/research_crew/crew.py
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool
from tool_choice.tools.custom_tool import add_numbers


serperd_tool = SerperDevTool(n_results=3)
sum_tool = add_numbers

@CrewBase
class ToolSearchCrew():
    """Crew for web search and sum tasks, as defined in agents.yaml and tasks.yaml (English version)"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def web_searcher(self) -> Agent:
        return Agent(
            config=self.agents_config['web_searcher'],
            verbose=True,
            tools=[serperd_tool]
        )

    @agent
    def summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['summarizer'],
            verbose=True
        )

    @agent
    def calculator(self) -> Agent:
        return Agent(
            config=self.agents_config['calculator'],
            verbose=True,
            tools=[sum_tool]
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

    @task
    def sum_task(self) -> Task:
        return Task(
            config=self.tasks_config['sum_task']
        )

    @crew
    def web_search_crew(self) -> Crew:
        """Crew for web search and summarization pipeline"""
        return Crew(
            agents=[
                self.web_searcher(),
                self.summarizer()
            ],
            tasks=[
                self.web_search_task(),
                self.summarization_task()
            ],
            process=Process.sequential,
            verbose=True,
        )

    @crew
    def sum_crew(self) -> Crew:
        """Crew for sum calculation pipeline"""
        return Crew(
            agents=[
                self.calculator()
            ],
            tasks=[
                self.sum_task()
            ],
            process=Process.sequential,
            verbose=True,
        )