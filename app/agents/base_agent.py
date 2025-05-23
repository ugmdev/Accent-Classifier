import abc
from typing import Dict, Any, Optional

class Agent(abc.ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, name: str):
        self.name = name
        self.messages = []
        
    def receive_message(self, message: Dict[str, Any], sender: Optional[str] = None):
        """Receive a message from another agent or the system"""
        self.messages.append({
            "sender": sender or "system",
            "content": message
        })
        return self.process_message(message, sender)
    
    @abc.abstractmethod
    def process_message(self, message: Dict[str, Any], sender: Optional[str] = None) -> Dict[str, Any]:
        """Process the received message and take appropriate action"""
        pass
    
    def send_message(self, recipient, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to another agent"""
        if hasattr(recipient, 'receive_message'):
            return recipient.receive_message(message, sender=self.name)
        else:
            raise ValueError(f"Recipient {recipient} cannot receive messages")
    
    def get_last_message(self) -> Optional[Dict[str, Any]]:
        """Get the last message received"""
        if self.messages:
            return self.messages[-1]
        return None