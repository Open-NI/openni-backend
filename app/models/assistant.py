from enum import Enum

class AssistantVoice(Enum):
    """Enum for different assistant voices and their personality traits."""
    
    AF_HEART = "af_heart"
    AM_MICHAEL = "am_michael"
    AF_PROFESSIONAL = "af_professional"
    AF_FRIENDLY = "af_friendly"
    AF_HUMOROUS = "af_humorous"
    
    @classmethod
    def get_personality_trait(cls, voice: str) -> str:
        """
        Get the personality trait for a given voice.
        
        Args:
            voice: The voice identifier
            
        Returns:
            str: The personality trait description
        """
        try:
            voice_enum = cls(voice)
            return cls._get_trait_for_voice(voice_enum)
        except ValueError:
            # Default to a standard personality if voice not found
            return "You are a helpful assistant. Your responses MUST be brief and concise."
    
    @staticmethod
    def _get_trait_for_voice(voice: 'AssistantVoice') -> str:
        """
        Get the personality trait for a specific voice enum.
        
        Args:
            voice: The voice enum
            
        Returns:
            str: The personality trait description
        """
        traits = {
            AssistantVoice.AF_HEART: "You are a sassy assistant with attitude. Your responses MUST be brief, concise, and include a touch of sass and humor, do not include emojis.",
            AssistantVoice.AF_PROFESSIONAL: "You are a professional assistant with a formal tone. Your responses MUST be brief, concise, and business-like, do not include emojis.",
            AssistantVoice.AF_FRIENDLY: "You are a friendly assistant with a casual, approachable tone. Your responses MUST be brief, concise, and conversational, do not include emojis.",
            AssistantVoice.AF_HUMOROUS: "You are a humorous assistant with a playful personality. Your responses MUST be brief, concise, and include a touch of wit, do not include emojis.",
            AssistantVoice.AM_MICHAEL: "You are a friendly assistant with a casual, approachable tone, but you keep mentioning that your name is Michael, everwhere. Your responses MUST be brief, concise, and conversationa, do not include emojis."
        }
        return traits.get(voice, "You are a helpful assistant. Your responses MUST be brief and concise.") 