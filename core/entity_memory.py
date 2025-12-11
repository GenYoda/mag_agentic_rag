"""
Entity Memory for Agentic Medical RAG

Tracks medical entities across conversation:
- PERSON: Patient names, doctor names
- MEDICATION: Drug names, dosages
- DIAGNOSIS: Medical conditions, diseases
- TEST: Lab tests, procedures
- DATE: Dates mentioned
- MEASUREMENT: Vital signs, lab values

Supports:
- LLM-based extraction (primary)
- Regex-based extraction (fallback)
- Entity history tracking
- Coreference resolution
"""

import re
import json
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
from collections import defaultdict

from utils.azure_clients import get_chat_completion
from utils.prompt_templates import get_entity_extraction_prompts


# ============================================================================
# SECTION 1: Entity Type Definitions
# ============================================================================

ENTITY_TYPES = [
    "PERSON",       # Patient names, doctor names
    "MEDICATION",   # Drugs, dosages
    "DIAGNOSIS",    # Conditions, diseases
    "TEST",         # Lab tests, procedures
    "DATE",         # Dates mentioned
    "MEASUREMENT"   # Vital signs, lab values
]


# ============================================================================
# SECTION 2: Regex-Based Entity Extraction (Fallback)
# ============================================================================

def extract_medications_regex(text: str) -> List[str]:
    """
    Extract medication mentions using regex patterns.
    
    Patterns:
    - Common drug suffixes: -statin, -pril, -olol, -cillin
    - Dosage patterns: "XXmg", "XX mg"
    
    Args:
        text: Input text
        
    Returns:
        List of medication mentions
    """
    medications = set()
    
    # Common medication suffixes
    patterns = [
        r'\b\w+(?:statin|pril|olol|cillin|mycin|cycline|azole|pine)\b',  # Drug suffixes
        r'\b[A-Z][a-z]+(?:formin|gliptin|sartan)\b',  # More patterns
        r'\b(?:aspirin|insulin|warfarin|heparin|morphine|codeine)\b',  # Common drugs
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        medications.update(match.group() for match in matches)
    
    # Look for dosage patterns nearby (e.g., "500mg")
    dosage_pattern = r'\b(\w+)\s*\d+\s*mg\b'
    matches = re.finditer(dosage_pattern, text, re.IGNORECASE)
    medications.update(match.group(1) for match in matches)
    
    return list(medications)


def extract_measurements_regex(text: str) -> List[str]:
    """
    Extract vital signs and measurements using regex.
    
    Patterns:
    - Blood pressure: "120/80", "BP 120/80"
    - Glucose: "glucose 150", "150 mg/dL"
    - Temperature: "98.6°F", "37°C"
    - Heart rate: "HR 88", "88 bpm"
    
    Args:
        text: Input text
        
    Returns:
        List of measurement mentions
    """
    measurements = set()
    
    patterns = [
        r'\b(?:BP|blood pressure)[\s:]*(\d{2,3}/\d{2,3})\b',  # Blood pressure
        r'\b(?:glucose|sugar)[\s:]*(\d{2,3})\s*(?:mg/dL)?\b',  # Glucose
        r'\b(?:temp|temperature)[\s:]*(\d{2,3}(?:\.\d)?)\s*[°]?[CF]\b',  # Temperature
        r'\b(?:HR|heart rate)[\s:]*(\d{2,3})\s*(?:bpm)?\b',  # Heart rate
        r'\b(\d{2,3}/\d{2,3})\s*(?:mmHg)?\b',  # Generic BP
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        measurements.update(match.group() for match in matches)
    
    return list(measurements)


def extract_dates_regex(text: str) -> List[str]:
    """
    Extract dates using regex patterns.
    
    Patterns:
    - MM/DD/YYYY, MM-DD-YYYY
    - Month Day, Year
    - Relative: "today", "yesterday", "last week"
    
    Args:
        text: Input text
        
    Returns:
        List of date mentions
    """
    dates = set()
    
    patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',  # Month Day, Year
        r'\b(?:today|yesterday|tomorrow)\b',  # Relative dates
        r'\b(?:last|next)\s+(?:week|month|year)\b',  # Relative periods
    ]
    
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        dates.update(match.group() for match in matches)
    
    return list(dates)


def extract_entities_regex(text: str) -> Dict[str, List[str]]:
    """
    Extract all entity types using regex (fallback method).
    
    Args:
        text: Input text
        
    Returns:
        Dict mapping entity type to list of mentions
    """
    return {
        "PERSON": [],  # Hard to extract reliably with regex
        "MEDICATION": extract_medications_regex(text),
        "DIAGNOSIS": [],  # Hard to extract reliably with regex
        "TEST": [],  # Hard to extract reliably with regex
        "DATE": extract_dates_regex(text),
        "MEASUREMENT": extract_measurements_regex(text)
    }


# ============================================================================
# SECTION 3: LLM-Based Entity Extraction (Primary)
# ============================================================================

def extract_entities_llm(text: str) -> Dict[str, List[str]]:
    """
    Extract entities using LLM.
    
    Uses prompt from utils/prompt_templates.py for medical entity extraction.
    
    Args:
        text: Input text
        
    Returns:
        Dict mapping entity type to list of mentions
    """
    try:
        system_prompt, user_prompt = get_entity_extraction_prompts(text)
        
        response = get_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for consistent extraction
            max_tokens=500
        )
        
        # Parse JSON response
        entities = json.loads(response)
        
        # Validate structure
        for entity_type in ENTITY_TYPES:
            if entity_type not in entities:
                entities[entity_type] = []
        
        return entities
        
    except Exception as e:
        print(f"⚠️  LLM extraction failed: {e}, falling back to regex")
        return extract_entities_regex(text)


def extract_entities(text: str, use_llm: bool = True) -> Dict[str, List[str]]:
    """
    Extract entities from text.
    
    Primary method: LLM-based extraction
    Fallback: Regex-based extraction
    
    Args:
        text: Input text
        use_llm: Use LLM for extraction (default: True)
        
    Returns:
        Dict mapping entity type to list of mentions
    """
    if use_llm:
        return extract_entities_llm(text)
    else:
        return extract_entities_regex(text)


# ============================================================================
# SECTION 4: Entity Memory Class
# ============================================================================

class EntityMemory:
    """
    Tracks entities mentioned across conversation.
    
    Features:
    - Store entities by type
    - Track when entities were mentioned
    - Retrieve recent entities
    - Support coreference resolution
    """
    
    def __init__(self, max_history: int = 50):
        """
        Initialize entity memory.
        
        Args:
            max_history: Maximum number of entity mentions to keep per type
        """
        self.max_history = max_history
        
        # Storage: {entity_type: [(entity, timestamp, context)]}
        self.entities: Dict[str, List[tuple]] = defaultdict(list)
        
        # Quick lookup: {entity_type: Set[entity]}
        self.entity_sets: Dict[str, Set[str]] = defaultdict(set)
    
    def add_entities(self, entities: Dict[str, List[str]], context: str = ""):
        """
        Add entities to memory.
        
        Args:
            entities: Dict mapping entity type to list of mentions
            context: Optional context (e.g., the full question/answer)
        """
        timestamp = datetime.now().isoformat()
        
        for entity_type, mentions in entities.items():
            if entity_type not in ENTITY_TYPES:
                continue
            
            for mention in mentions:
                if not mention or not mention.strip():
                    continue
                
                # Add to history
                self.entities[entity_type].append((mention, timestamp, context))
                
                # Add to quick lookup
                self.entity_sets[entity_type].add(mention.lower())
                
                # Trim history if needed
                if len(self.entities[entity_type]) > self.max_history:
                    oldest = self.entities[entity_type].pop(0)
                    # Note: entity_sets is not trimmed (keeps all unique entities)
    
    def get_entities(self, entity_type: str, limit: int = 10) -> List[str]:
        """
        Get recent entities of a specific type.
        
        Args:
            entity_type: Type of entity to retrieve
            limit: Maximum number to return (most recent)
            
        Returns:
            List of entity mentions (most recent first)
        """
        if entity_type not in self.entities:
            return []
        
        # Get most recent entities
        recent = self.entities[entity_type][-limit:]
        return [entity for entity, _, _ in reversed(recent)]
    
    def get_all_entities(self, limit_per_type: int = 10) -> Dict[str, List[str]]:
        """
        Get recent entities of all types.
        
        Args:
            limit_per_type: Maximum per entity type
            
        Returns:
            Dict mapping entity type to list of mentions
        """
        return {
            entity_type: self.get_entities(entity_type, limit=limit_per_type)
            for entity_type in ENTITY_TYPES
        }
    
    def has_entity(self, entity_type: str, entity: str) -> bool:
        """
        Check if entity was mentioned.
        
        Args:
            entity_type: Type of entity
            entity: Entity mention (case-insensitive)
            
        Returns:
            bool: True if entity was mentioned
        """
        return entity.lower() in self.entity_sets.get(entity_type, set())
    
    def get_entity_context(self, entity_type: str, entity: str) -> List[str]:
        """
        Get contexts where entity was mentioned.
        
        Args:
            entity_type: Type of entity
            entity: Entity mention
            
        Returns:
            List of context strings
        """
        if entity_type not in self.entities:
            return []
        
        contexts = []
        for mention, _, context in self.entities[entity_type]:
            if mention.lower() == entity.lower() and context:
                contexts.append(context)
        
        return contexts
    
    def get_recent_context(self, max_entities: int = 5) -> str:
        """
        Get formatted string of recent entities for context.
        
        Used for coreference resolution and context-aware responses.
        
        Args:
            max_entities: Maximum entities to include per type
            
        Returns:
            Formatted string listing recent entities
        """
        context_parts = []
        
        for entity_type in ENTITY_TYPES:
            entities = self.get_entities(entity_type, limit=max_entities)
            if entities:
                context_parts.append(f"{entity_type}: {', '.join(entities)}")
        
        if not context_parts:
            return "No entities tracked yet."
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear all entity memory."""
        self.entities.clear()
        self.entity_sets.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get entity memory statistics.
        
        Returns:
            Dict with statistics
        """
        return {
            "total_mentions": sum(len(mentions) for mentions in self.entities.values()),
            "unique_entities": sum(len(entities) for entities in self.entity_sets.values()),
            "by_type": {
                entity_type: {
                    "mentions": len(self.entities.get(entity_type, [])),
                    "unique": len(self.entity_sets.get(entity_type, set()))
                }
                for entity_type in ENTITY_TYPES
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize entity memory to dictionary.
        
        Returns:
            Dict representation
        """
        return {
            "entities": {
                entity_type: [(e, t, c) for e, t, c in mentions]
                for entity_type, mentions in self.entities.items()
            },
            "max_history": self.max_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityMemory':
        """
        Deserialize entity memory from dictionary.
        
        Args:
            data: Dict representation
            
        Returns:
            EntityMemory instance
        """
        memory = cls(max_history=data.get("max_history", 50))
        
        for entity_type, mentions in data.get("entities", {}).items():
            for entity, timestamp, context in mentions:
                memory.entities[entity_type].append((entity, timestamp, context))
                memory.entity_sets[entity_type].add(entity.lower())
        
        return memory


# ============================================================================
# SECTION 5: Helper Functions
# ============================================================================

def extract_and_store(
    text: str,
    entity_memory: EntityMemory,
    context: str = "",
    use_llm: bool = True
) -> Dict[str, List[str]]:
    """
    Extract entities and store in memory (convenience function).
    
    Args:
        text: Text to extract from
        entity_memory: EntityMemory instance
        context: Optional context string
        use_llm: Use LLM for extraction
        
    Returns:
        Extracted entities dict
    """
    entities = extract_entities(text, use_llm=use_llm)
    entity_memory.add_entities(entities, context=context)
    return entities


def print_entity_memory(entity_memory: EntityMemory):
    """
    Print formatted entity memory contents.
    
    Args:
        entity_memory: EntityMemory instance
    """
    print("\n" + "=" * 80)
    print("ENTITY MEMORY")
    print("=" * 80)
    
    stats = entity_memory.get_stats()
    print(f"\nTotal Mentions: {stats['total_mentions']}")
    print(f"Unique Entities: {stats['unique_entities']}")
    
    print("\nBy Type:")
    for entity_type in ENTITY_TYPES:
        type_stats = stats['by_type'][entity_type]
        if type_stats['mentions'] > 0:
            entities = entity_memory.get_entities(entity_type, limit=5)
            print(f"\n  {entity_type}:")
            print(f"    • Mentions: {type_stats['mentions']}")
            print(f"    • Unique: {type_stats['unique']}")
            print(f"    • Recent: {', '.join(entities[:3])}")
    
    print("\n" + "=" * 80 + "\n")
