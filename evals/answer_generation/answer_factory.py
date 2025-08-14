from abc import abstractmethod
from datetime import datetime
import json
import os
from answer_generation.answer_generator import AnswerGenerator

class AnswerFactory:
    def __init__(
            self, 
            answer_generator: AnswerGenerator, 
            input_file_path: str,
            output_folder_path: str
        ):
        self.answer_generator = answer_generator
        self.input_file_path = input_file_path
        self.output_folder_path = output_folder_path
        self.output_file_path = self._create_output_file()

    @abstractmethod
    def run(self) -> str:
        pass

    @abstractmethod
    def _load_input_data(self):
        pass
    
    def _append_to_output_file(self, content: json):
        # Append to the JSONL file
        with open(self.output_file_path, 'a', encoding='utf-8') as file:
            json.dump(content, file)
            file.write('\n')

    # Use jsonl file since it is more efficient for appending data
    def _create_output_file(self):
        # Create an output file in the specified folder path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = f"{self.output_folder_path}/chatbot_answers_{timestamp}.jsonl"
        
        # Check if file already exists
        if os.path.exists(output_file_path):
            raise FileExistsError(output_file_path)

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        # Create an empty JSONL file
        with open(output_file_path, 'w', encoding='utf-8') as file:
            pass  # Create empty file
        return output_file_path
