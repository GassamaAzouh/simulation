from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import yaml
import os
from simulation.tools.custom_tool import DeleteFile_Bank_Tool,DeleteFile_Ref_Tool,DeleteFile_Rule_Tool
from crewai.memory import LongTermMemory


# , CrewBase_2

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


gemini = LLM(
    model='gemini/gemini-1.5-flash',
    #model = 'gemini/gemini-2.0-flash',
    temperature=0.7,
	api_key= os.getenv('GEMINI_API_KEY')
)
groq_llama_3 = LLM(
    model='groq/llama3-70b-8192',
    temperature=0.7,
	api_key= os.getenv('GROQ_API_KEY')
)
groq_llama = LLM(
    model = "groq/llama-3.3-70b-versatile",
    #model = "groq/lama3-8b-8192",
    temperature=0.9,
	api_key= os.getenv('GROQ_API_KEY')
)

grog_mixtral = LLM(
    model = "groq/mixtral-8x7b-32768",
    temperature=0.9,
	api_key= os.getenv('GROQ_API_KEY')
)
memory_trans = LongTermMemory( path="memory_trans.csv")
memory_rules = LongTermMemory(path="memory_rules.txt")
memory_new_trans = LongTermMemory( path="memory_new_trans.csv")
#memory_trans_classify = LongTermMemory( path="memory_trans_classify.csv")


@CrewBase
class Simulation():
	"""Simulation crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	
	@agent
	def transaction_generator_agent(self) -> Agent:
		return Agent(
			config= self.agents_config["transaction_generator_agent"],
			#config=self.agents_config["transaction_generator_agent"],
			verbose=True,
			#max_rpm= 3 ,
			llm=groq_llama_3,
			memory = memory_trans
			#tools= [DeleteFile_Ref_Tool()]
		)
	
	@agent
	def rules_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['rules_agent'],
			verbose=True,
			#max_rpm= 3 ,
			llm=gemini,
			memory = memory_rules
			#tools= [DeleteFile_Rule_Tool()]
		)
	
	@agent
	def generator_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['generator_agent'],
			verbose=True,
			#max_rpm= 3 ,
			llm=groq_llama,
			memory=memory_new_trans
		)
	
	@agent
	def transaction_classifier_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['transaction_classifier_agent'],
			verbose=True,
			#max_rpm= 3 ,
			llm=gemini,
			#memory=memory_trans_classify
			#tools= [DeleteFile_Bank_Tool()]
		)
	
	
	@task
	def generator_task(self) -> Task:
		return Task(
			config= self.tasks_config['generator_task'] ,
			output_file='Transactions_references.csv',
			#max_rpm = 3 ,
		)
	
	@task
	def ruler_task(self) -> Task:
		return Task(
			config=self.tasks_config['ruler_task'],
			output_file='Regles.txt',
			context= [self.generator_task(), ],
			#max_rpm = 3 ,
		)
	
	@task
	def new_generator_task(self) -> Task:
		return Task(
			config=self.tasks_config['new_generator_task'],
			output_file='New_transaction.csv',
			context= [self.generator_task(), ],
			#max_rpm = 3 ,
		)
	
	@task
	def classify_task(self) -> Task:
		return Task(
			config=self.tasks_config['classify_task'],
			output_file='Bank_transaction.csv',
			context= [self.ruler_task(), self.new_generator_task(), ],
			#max_rpm = 3 ,
		)
	
	
	@crew
	def crew(self) -> Crew:
		"""Creates the Simulation crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents , # Automatically created by the @agent decorator
			#agents=[self.transaction_generator_agent, self.rules_agent, self.generator_agent, self.transaction_classifier_agent] ,
			tasks=self.tasks, # Automatically created by the @task decorator
			#tasks=[self.generator_task, self.ruler_task, self.new_generator_task, self.classify_task] ,
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
"""	
	@agent
	def client_behavior_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['client_behavior_agent'],
			verbose=True,
			#max_rpm= 3 ,
			#llm=gemini
		)
	
	@agent
	def normal_transaction_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['normal_transaction_agent'],
			verbose=True,
			#max_rpm= 3 ,
			#llm=gemini
		)
	
	@agent
	def fraude_transaction_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['fraude_transaction_agent'],
			verbose=True,
			#max_rpm= 3 ,
			#llm=gemini
		)
	
	@task
	def generate_clients_task(self) -> Task:
		return Task(
			config=self.tasks_config['generate_clients_task'],
			
			output_file = "Clients.txt",
			#max_rpm = 3 ,
		)
	
	@task
	def normal_transactions_task(self) -> Task:
		return Task(
			config=self.tasks_config['normal_transactions_task'],
			output_file='Normales.csv',
			context= [self.generate_clients_task(), ],
			#max_rpm = 3 ,
		)
	
	@task
	def fraude_transactions_task(self) -> Task:
		return Task(
			config=self.tasks_config['fraude_transactions_task'],
			output_file='Fraude.csv',
			context= [self.generate_clients_task(), self.fraude_transactions_task()],
			#max_rpm = 3 ,
		)
	

	@crew
	def crew_2(self) -> Crew:

		return Crew(
			#agents=self.agents ,
			agents= [self.client_behavior_agent, self.normal_transaction_agent, self.fraude_transaction_agent] ,
			#tasks=self.tasks , 
			tasks= [self.generate_clients_task, self.normal_transactions_task, self.fraude_transactions_task] ,
			process=Process.sequential,
			verbose=True,
			respect_context_window=True
			)
	
	def scenario2(self, input) :
		return self.crew_2().kickoff(inputs= input)
"""	