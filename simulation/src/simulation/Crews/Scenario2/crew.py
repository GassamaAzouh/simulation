from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import yaml
import os
from simulation.tools.custom_tool import DeleteFile_Bank_Tool,DeleteFile_Ref_Tool,DeleteFile_Rule_Tool
from crewai.memory import LongTermMemory

# Création d'une mémoire persistante
memory_comportement = LongTermMemory( path="memory_commportement.txt")
memory_trans_normal = LongTermMemory( path="memory_trans_normal.csv")
memory_trans_fraude = LongTermMemory( path="memory_trans_fraude.csv")
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


@CrewBase
class simulation() :
#C:/Users/JJ/OneDrive/Desktop/Mem/simulation/src/simulation/

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	#print(agents_config["normal_transaction_agent"])
	#print(f"Type de tasks_config: {type(tasks_config)}")

	@agent
	def client_behavior_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['client_behavior_agent'],
			verbose=True,
			max_rpm= 3 ,
			llm=groq_llama_3,
			memory=memory_comportement
		)
	
	@agent
	def normal_transaction_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['normal_transaction_agent'],
			verbose=True,
			max_rpm= 3 ,
			llm=groq_llama ,
			memory=memory_trans_normal
		)
	
	@agent
	def fraude_transaction_agent(self) -> Agent:
		return Agent(
			config=self.agents_config['fraude_transaction_agent'],
			verbose=True,
			max_rpm= 3 ,
			llm=gemini, 
			memory=memory_trans_fraude
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
			context= [self.generate_clients_task(), self.normal_transactions_task()],
			#max_rpm = 3 ,
		)
	

	@crew
	def crew(self) -> Crew:

		return Crew(
			agents=self.agents ,
			tasks=self.tasks , 
			process=Process.sequential,
			verbose=True,
			#respect_context_window=True
			)