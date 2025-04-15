from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import os


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."

class DeleteFile_Ref_Tool(BaseTool):
    name : str = "Supprimer l'ancien fichier de références"
    description : str = "Supprime le fichier nommé 'Transaction_references.csv' s'il existe dans l'environnement de travail. Cela permet d'éviter de garder tout les données simulées de tout les exécutions. Pour chaque nouvelle exécution, l'ancien fichier de référence sera supprimé"

    def _run(self):
        file_path = "Transaction_references.csv"
        if os.path.exists(file_path):
            os.remove(file_path)
            return f"Le fichier '{file_path}' a été supprimé avec succès."
        else:
            return f"Le fichier '{file_path}' n'existe pas."
        
class DeleteFile_Rule_Tool(BaseTool):
    name : str = "Supprimer l'ancien fichier des règles de détection"
    description : str = "Supprime le fichier nommé 'Regles.txt' s'il existe dans l'environnement de travail. Cela permet d'éviter de garder tout les données simulées de tout les exécutions. Pour chaque nouvelle exécution, l'ancien fichier des règles sera supprimé"

    def _run(self):
        file_path = "Regles.txt"
        if os.path.exists(file_path):
            os.remove(file_path)
            return f"Le fichier '{file_path}' a été supprimé avec succès."
        else:
            return f"Le fichier '{file_path}' n'existe pas."
        
class DeleteFile_Bank_Tool(BaseTool):
    name : str = "Supprimer l'ancien fichier des données bancaires"
    description : str = "Supprime le fichier nommé 'Bank_transaction.csv' s'il existe dans l'environnement de travail. Cela permet d'éviter de garder tout les données simulées de tout les exécutions. Pour chaque nouvelle exécution, l'ancien fichier des données bancaires sera supprimé"

    def _run(self):
        file_path = "Bank_transaction.csv"
        if os.path.exists(file_path):
            os.remove(file_path)
            return f"Le fichier '{file_path}' a été supprimé avec succès."
        else:
            return f"Le fichier '{file_path}' n'existe pas."