import os
from typing import Any, List, Optional


from langchain_core.outputs import ChatGeneration, ChatResult
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain.schema import SystemMessage as LangChainSystemMessage

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
import dotenv

dotenv.load_dotenv()

class AzureChatModel(BaseChatModel):
    """Custom LangChain chat model wrapper for Azure AI Inference SDK"""
    
    # Declare Pydantic fields
    client: Any = None
    model_name: str
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like ChatCompletionsClient

    def __init__(
            self, 
            endpoint: str = os.getenv("PROJECT_ENDPOINT") + os.getenv("LLM_MODEL"), 
            api_key: str = os.getenv("ENDPOINT_KEY"), 
            model_name: str = os.getenv("LLM_MODEL")
        ):
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
        
        # Pass the required fields to the parent constructor
        super().__init__(client=client, model_name=model_name)
    
    def _generate(
            self, 
            messages: List[BaseMessage], 
            stop: Optional[List[str]] = None,
            run_manager: Optional[Any] = None,
            **kwargs
        ) -> ChatResult:
        # Convert LangChain messages to Azure AI Inference format
        azure_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                azure_messages.append(UserMessage(content=msg.content))
            elif isinstance(msg, LangChainSystemMessage):
                azure_messages.append(SystemMessage(content=msg.content))
            elif isinstance(msg, AIMessage):
                azure_messages.append(UserMessage(content=msg.content))  # Treat as user for context
        
        try:
            response = self.client.complete(
                messages=azure_messages,
                model=self.model_name,
                **kwargs
            )
            
            # Extract the response content
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                
                # Create proper ChatResult structure
                message = AIMessage(content=content)
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
            else:
                raise RuntimeError("No response from Azure model")
                
        except Exception as e:
            raise RuntimeError(f"Error calling Azure model: {str(e)}")
    
    def _llm_type(self) -> str:
        return "azure_inference"
    
    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name}