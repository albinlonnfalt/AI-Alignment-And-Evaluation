import json
from datetime import datetime
import requests
from .auth_service import AuthService
from .chat_session_initializer import ChatSessionInitializer

class AnswerGenerator:
    def __init__(
            self,
            base_url: str,
            auth_service: AuthService,
            expert_identifier: str
        ):
        self.base_url = base_url
        self.auth_service = auth_service
        self.expert_identifier = expert_identifier
        self.chat_session_initializer = ChatSessionInitializer(
            base_url=base_url,
            expert_identifier=expert_identifier,
            auth_service=auth_service
        )

    def create_answer(self, question):
        
        # Create a new chat session
        chat_id = self._create_new_chat_session()
        
        # Build the payload and headers
        headers = self._get_headers()
        payload = self._get_payload(question, chat_id)

        # Build full URL
        full_url = self._build_full_url(chat_id)

        response = requests.post(
            full_url,
            headers=headers,
            json=payload,
            verify=False
        )

        # Check if request was successful
        if response.status_code == 200:
            response_data = response.json()
            
            # Extract the chatbot response from the variables
            if 'variables' in response_data:
                for variable in response_data['variables']:
                    if variable.get('key') == 'input':
                        return variable.get('value', '')
            
            # If 'input' key not found, return the full response for debugging
            print("Full response:", response_data)
            return "Error: Could not extract response from chatbot"
        else:
            print(f"Error: API request failed with status code {response.status_code}")
            print("Response:", response.text)
            return f"Error: API request failed with status code {response.status_code}"


    def _get_headers(self):
        return self.auth_service.get_auth_headers()
    
    def _get_payload(self, question, chat_id):
        
        history = {
            "Id": "",
            "UserId": "e110e23e-d5a4-45dc-b29b-3224d98aa664.33dab507-5210-4075-805b-f2717d8cfa74",
            "UserName": "Tester Testsson",
            "ChatId": chat_id,
            "Content": question,
            "Type": 0,
            "AuthorRole": 0,
            "UserFeedback": 0,
            "Timestamp": datetime.now().isoformat()
        } 
        
        return {
            "input": question,
            "variables": [
                {"key": "chatId", "value": chat_id},
                {"key": "messageType", "value": "0"},
                {"key": "history", "value": json.dumps([history])},
                {"key": "citationIds", "value": None},
                {"key": "optimizeToken", "value": "true"}
            ]
        }
    

    def _build_full_url(self, chatid):
        return f"{self.base_url}/chats/{chatid}/messages?includeDiagnosticTraces=false"

    def _create_new_chat_session(self):
        "Return the session ID of the newly created chat session."
        return self.chat_session_initializer.initialize_chat_session()