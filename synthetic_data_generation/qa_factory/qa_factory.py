import json
import os
from datetime import datetime
from typing import List, Tuple
from generators import BaseGenerator
from tqdm import tqdm
from tracing.telemetry import get_tracer, traced

class QAFactory:
    def __init__(
            self, 
            generators: List[Tuple[BaseGenerator, float]],
            output_folder: str = "data/q-a"
            ):
        """
        Initialize QAFactory with generators and their usage percentages.
        
        Args:
            generators: List of tuples where each tuple contains:
                - BaseGenerator instance
                - float representing the percentage (0.0 to 1.0) of how much this generator should be used
        
        Example:
            factory = QAFactory([
                (generator1, 0.6),  # 60% usage
                (generator2, 0.4)   # 40% usage
            ])
        """
        # Validate that percentages sum to 1.0 (with small tolerance for floating point precision)
        total_percentage = sum(percentage for _, percentage in generators)
        if abs(total_percentage - 1.0) > 0.01:
            raise ValueError(f"Generator percentages must sum to 1.0, got {total_percentage}")
        
        # Validate that all percentages are between 0 and 1
        for generator, percentage in generators:
            if not (0.0 <= percentage <= 1.0):
                raise ValueError(f"Percentage must be between 0.0 and 1.0, got {percentage}")
        
        self.generators = generators
        self.output_folder = output_folder
        self.tracer = get_tracer("qa-factory")

    @traced("qa_factory.generate_samples")
    def generate(self, number_of_samples: int) -> Tuple[List[dict], str]:
        """
        Generate a list of Q&A samples based on the configured generators and their usage percentages.
        
        Args:
            number_of_samples: Total number of Q&A samples to generate
            
        Returns:
            Tuple[List[dict], str]: Tuple containing list of generated Q&A samples and the file path
        """
        samples = []
        
        # Create a single progress bar for all samples
        with tqdm(total=number_of_samples, desc="Generating Q&A samples", unit="samples") as pbar:
            for i, (generator, percentage) in enumerate(self.generators):
                # Calculate how many samples this generator should produce
                num_samples_for_generator = int(number_of_samples * percentage)
                
                if num_samples_for_generator == 0:
                    continue
                
                # Create a span for each generator's batch to ensure proper nesting
                with self.tracer.start_as_current_span(f"qa_factory.generator_{generator.__class__.__name__}") as generator_span:
                    generator_span.set_attribute("generator.class", generator.__class__.__name__)
                    generator_span.set_attribute("generator.samples_to_generate", num_samples_for_generator)
                    generator_span.set_attribute("generator.percentage", percentage)
                    
                    # Generate the required number of samples from this generator
                    for sample_idx in range(num_samples_for_generator):
                        sample = generator.generate()
                        samples.append(sample)
                        pbar.update(1)  # Update the progress bar
        
        full_path = self.save_qa_to_json(samples)

        print(f"✓ Successfully generated {len(samples)} total Q&A samples")
        return samples, full_path

    def save_qa_to_json(self, qa_samples: List[any]):
        """
        Save the generated Q&A samples to a JSON file.
        
        Args:
            qa_samples: List of Q&A samples to save (QATagged objects)
        """

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qa_generated_{timestamp}.json"

        # Combine output_folder (folder) with generated filename
        full_path = os.path.join(self.output_folder, filename)

        # Ensure the directory exists
        os.makedirs(self.output_folder, exist_ok=True)

        # Convert QATagged objects to dictionaries for JSON serialization
        serializable_samples = [sample.model_dump() for sample in qa_samples]

        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_samples, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved {len(qa_samples)} Q&A samples to {full_path}")

        return full_path