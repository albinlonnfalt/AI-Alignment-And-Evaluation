import numpy as np
from models import LengthDistributionConfig, DiversityInjection


class DiversityGenerator:

    def __init__(self, config: LengthDistributionConfig):
        self.config = config

    def get_diversity_injection(self) -> DiversityInjection:

        return DiversityInjection(
            response_length=self._get_response_length(),
            tone_injection=self._get_tone_injection(),
            disruptive_injection=self._get_disruptive_injection(),
            language_injection=self._get_language_injection()
        )

    def get_injection_as_string(self, injection: DiversityInjection) -> str:
        return (
            f"Response Length: {injection.response_length}\n"
            f"Tone Injection: {injection.tone_injection}\n"
            f"Disruptive Injection: {injection.disruptive_injection}\n"
            f"Language Injection: {injection.language_injection}\n"
        )

    def _get_response_length(self) -> int:
        """
        Generate a response length using a log-normal distribution and a shift.
        """
        mean = self.config.mean
        sigma = self.config.sigma
        shift = self.config.shift

        length = np.random.lognormal(mean=mean, sigma=sigma) + shift
        return int(max(shift, length))
    
    def _get_tone_injection(self) -> str:
        """
        Generate a tone injection based on the diversity configuration.
        This uses weighted random selection based on frequency scores.
        """
        tones = [
            {"name": "", "frequency": 5.0},
            {"name": "formal", "frequency": 1.0},
            {"name": "informal", "frequency": 1.5},
            {"name": "conversational", "frequency": 2.0}
        ]
        
        tone_names = [tone["name"] for tone in tones]
        frequencies = [tone["frequency"] for tone in tones]
        
        return np.random.choice(tone_names, p=np.array(frequencies) / np.sum(frequencies))

    def _get_disruptive_injection(self) -> str:
        """
        Generate a disruptive injection based on the diversity configuration.
        This uses weighted random selection based on frequency scores.
        """
        disruptive_phrases = [
            {"name": "", "frequency": 5.0},
            {"name": "Include a few spelling errors in the question you generate.", "frequency": 1.5},
            {"name": "Introduce a slight ambiguity in the question you generate.", "frequency": 1.0}
        ]
        
        phrase_names = [phrase["name"] for phrase in disruptive_phrases]
        frequencies = [phrase["frequency"] for phrase in disruptive_phrases]
        
        return np.random.choice(phrase_names, p=np.array(frequencies) / np.sum(frequencies))
    
    def _get_language_injection(self) -> str:
        """
        Generate a language injection based on the diversity configuration.
        This uses weighted random selection based on frequency scores.
        """
        languages = [
            {"name": "English", "frequency": 10},
            {"name": "Swedish", "frequency": 1},
        ]
        
        language_names = [lang["name"] for lang in languages]
        frequencies = [lang["frequency"] for lang in languages]
        
        return np.random.choice(language_names, p=np.array(frequencies) / np.sum(frequencies))
