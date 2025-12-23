"""
================================================================================
VALIDATION TOOLS - Phase 3 Tool 9a/10 (UPDATED)
================================================================================
Purpose: Validate generated answers for quality, safety, and compliance
Features:
- Answer relevance validation (LLM-as-Judge)
- Hallucination detection (LLM-as-Judge)
- Citation validation (Rule-based)
- Quality scoring (LLM-as-Judge)
- NO PII DETECTION (returns unfiltered answers)
- Configurable LLM provider (Azure OpenAI → Gemini)
================================================================================
"""

import logging
import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from crewai.tools import tool
import re
from typing import List, Dict, Any

from datetime import datetime
import os

logger = logging.getLogger(__name__)

# ============================================================================
# TIER 1: Fast Rule-Based Validation (No LLM - Phase 2.1 Addition)
# ============================================================================

def quick_validate_answer(
    query: str,
    answer: str,
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Fast rule-based validation without LLM call.
    Catches obvious failures to avoid wasting LLM validation.
    """
    issues = []
    
    # Check 1: Citation presence
    citation_patterns = [
        r'\[doc:\d+\]',
        r'\[source:[^\]]+\]',
        r'\[page:\d+\]',
        r'\[\d+\]'
    ]
    has_citation = any(re.search(pattern, answer) for pattern in citation_patterns)
    
    if not has_citation:
        issues.append({
            'type': 'missing_citations',
            'severity': 'high',
            'description': 'Answer lacks source citations'
        })
    
    # Check 2: Citation validity
    if has_citation:
        doc_citations = re.findall(r'\[doc:(\d+)\]', answer)
        invalid_citations = []
        for doc_num in doc_citations:
            if int(doc_num) >= len(chunks):
                invalid_citations.append(f"doc:{doc_num}")
        
        if invalid_citations:
            issues.append({
                'type': 'invalid_citations',
                'severity': 'high',
                'description': f'Citations reference non-existent chunks: {invalid_citations}'
            })
    
    # Check 3: Uncertainty indicators
    uncertainty_phrases = [
        "i don't know", "i do not know", "not mentioned", "not clear",
        "unclear", "cannot determine", "not specified",
        "no information available", "unable to find"
    ]
    answer_lower = answer.lower()
    found_uncertainty = [phrase for phrase in uncertainty_phrases if phrase in answer_lower]
    
    if found_uncertainty:
        issues.append({
            'type': 'explicit_uncertainty',
            'severity': 'high',
            'description': f'Answer contains uncertainty: {found_uncertainty[0]}'
        })
    
    # Check 4: Extreme verbosity for simple questions
    word_count = len(answer.split())
    query_lower = query.lower()
    
    if any(q in query_lower for q in ['is it', 'is the', 'did the', 'was the', 'does the', 'can the']):
        if word_count > 100:
            issues.append({
                'type': 'suspiciously_verbose',
                'severity': 'medium',
                'description': f'Yes/No question has {word_count} word answer'
            })
    
    # Check 5: Too short
    if word_count < 3:
        issues.append({
            'type': 'too_short',
            'severity': 'medium',
            'description': f'Answer only {word_count} words'
        })
    
    # Decision
    high_severity_issues = [i for i in issues if i['severity'] == 'high']
    
    return {
        'quick_pass': len(high_severity_issues) == 0,
        'issues': issues,
        'total_issues': len(issues),
        'high_severity_count': len(high_severity_issues),
        'confidence': 'high' if len(issues) == 0 else 'low',
        'requires_llm_validation': True
    }


def should_skip_llm_validation(quick_result: Dict[str, Any]) -> bool:
    """
    Determine if we should skip LLM validation and go straight to self-heal.
    Only for obvious failures that don't need LLM diagnosis.
    """
    if not quick_result['quick_pass']:
        issue_types = [issue['type'] for issue in quick_result['issues']]
        skip_llm_issues = ['missing_citations', 'explicit_uncertainty', 'invalid_citations']
        
        if any(issue_type in skip_llm_issues for issue_type in issue_types):
            return True
    
    return False




# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ValidationIssue:
    """Represents a validation issue found"""
    severity: str  # 'critical', 'warning', 'info'
    category: str  # 'hallucination', 'citation', 'quality', 'safety'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    auto_fixable: bool = False


@dataclass
class ValidationResult:
    """Result of answer validation"""
    is_valid: bool
    quality_score: float  # 0.0 to 1.0
    confidence_score: float  # 0.0 to 1.0
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Specific checks
    has_hallucination: bool = False
    has_citations: bool = True
    citation_count: int = 0
    answer_length: int = 0
    
    # LLM-as-Judge results
    llm_judge_used: bool = False
    llm_judge_reasoning: str = ""
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    validation_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'is_valid': self.is_valid,
            'quality_score': self.quality_score,
            'confidence_score': self.confidence_score,
            'issues': [
                {
                    'severity': issue.severity,
                    'category': issue.category,
                    'message': issue.message,
                    'details': issue.details,
                    'auto_fixable': issue.auto_fixable
                }
                for issue in self.issues
            ],
            'warnings': self.warnings,
            'has_hallucination': self.has_hallucination,
            'has_citations': self.has_citations,
            'citation_count': self.citation_count,
            'answer_length': self.answer_length,
            'llm_judge_used': self.llm_judge_used,
            'timestamp': self.timestamp.isoformat(),
            'validation_time_ms': self.validation_time_ms
        }


# ============================================================================
# Validation Tools Class
# ============================================================================

class ValidationTools:
    """
    Validates generated answers for quality, safety, and compliance
    """
    
    def __init__(
        self,
        min_answer_length: int = 50,
        max_answer_length: int = 2000,
        min_citations: int = 1,
        min_quality_score: float = 0.6,
        enable_llm_judge: bool = True,
        llm_provider: str = "azure_openai",
        llm_deployment: str = None,
        fallback_to_rules: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize ValidationTools
        
        Args:
            min_answer_length: Minimum acceptable answer length
            max_answer_length: Maximum acceptable answer length
            min_citations: Minimum required citations
            min_quality_score: Minimum quality score threshold
            enable_llm_judge: Use LLM for validation (vs rule-based)
            llm_provider: LLM provider (azure_openai, gemini, openai)
            llm_deployment: Deployment name (Azure) or model name
            fallback_to_rules: Use rule-based if LLM fails
            strict_mode: Fail on any warning
        """
        self.min_answer_length = min_answer_length
        self.max_answer_length = max_answer_length
        self.min_citations = min_citations
        self.min_quality_score = min_quality_score
        self.enable_llm_judge = enable_llm_judge
        self.llm_provider = llm_provider
        self.llm_deployment = llm_deployment or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
        self.fallback_to_rules = fallback_to_rules
        self.strict_mode = strict_mode
        
        # Initialize LLM client if enabled
        self.llm_client = None
        if self.enable_llm_judge:
            self._initialize_llm_client()
        
        # Fallback: Rule-based hallucination indicators
        self.hallucination_phrases = [
            "i don't know",
            "i cannot",
            "i'm not sure",
            "there is no information",
            "the document does not mention",
            "according to my knowledge",
            "as an ai",
            "i apologize"
        ]
        
        # Quality indicators
        self.weak_phrases = [
            "maybe",
            "possibly",
            "might be",
            "could be",
            "perhaps",
            "it seems",
            "appears to be"
        ]
        
        logger.info(f"ValidationTools initialized (llm_judge: {enable_llm_judge}, "
                   f"provider: {llm_provider}, deployment: {self.llm_deployment})")
    
    
    def _initialize_llm_client(self):
        """Initialize LLM client based on provider"""
        try:
            if self.llm_provider == "azure_openai":
                from openai import AzureOpenAI
                
                self.llm_client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                )
                
                logger.info(f"✅ Azure OpenAI client initialized (deployment: {self.llm_deployment})")
            
            elif self.llm_provider == "gemini":
                logger.warning("Gemini provider not yet implemented, falling back to rules")
                self.llm_client = None
            
            elif self.llm_provider == "openai":
                from openai import OpenAI
                self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                logger.info("✅ OpenAI client initialized")
            
            else:
                logger.warning(f"Unknown provider: {self.llm_provider}, using rule-based")
                self.llm_client = None
        
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None
    
    
    # ========================================================================
    # Main Validation Method
    # ========================================================================
    
    def validate_answer(
        self,
        answer: str,
        query: str,
        context_chunks: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate a generated answer
        
        Args:
            answer: Generated answer to validate
            query: Original user query
            context_chunks: Retrieved context chunks used for generation
            metadata: Additional metadata
            
        Returns:
            ValidationResult object
        """
        start_time = datetime.now()
        
        result = ValidationResult(
            is_valid=True,
            quality_score=1.0,
            confidence_score=1.0,
            answer_length=len(answer)
        )
        
        # 1. Basic structural validation (rule-based, fast)
        self._validate_structure(answer, result)
        
        # 2. Citation validation (rule-based)
        self._validate_citations(answer, result)
        
        # 3. LLM-as-Judge validation (if enabled)
        if self.enable_llm_judge and self.llm_client:
            try:
                self._llm_judge_validate(answer, query, context_chunks, result)
                result.llm_judge_used = True
            except Exception as e:
                logger.error(f"LLM judge failed: {e}")
                if self.fallback_to_rules:
                    logger.info("Falling back to rule-based validation")
                    self._rule_based_validation(answer, query, context_chunks, result)
                else:
                    raise
        else:
            # Use rule-based validation
            self._rule_based_validation(answer, query, context_chunks, result)
        
        # 4. Safety checks (minimal - no PII)
        self._check_safety(answer, result)
        
        # Calculate overall validity
        critical_issues = [i for i in result.issues if i.severity == 'critical']
        result.is_valid = len(critical_issues) == 0
        
        if self.strict_mode:
            result.is_valid = result.is_valid and len(result.issues) == 0
        
        # Calculate validation time
        end_time = datetime.now()
        result.validation_time_ms = (end_time - start_time).total_seconds() * 1000
        
        logger.info(f"Validation complete: valid={result.is_valid}, "
                   f"quality={result.quality_score:.2f}, "
                   f"llm_judge={result.llm_judge_used}, "
                   f"issues={len(result.issues)}")
        
        return result
    
    
    # ========================================================================
    # Rule-Based Validation Components
    # ========================================================================
    
    def _validate_structure(self, answer: str, result: ValidationResult):
        """Validate basic answer structure"""
        
        # Check if answer is empty
        if not answer or not answer.strip():
            result.issues.append(ValidationIssue(
                severity='critical',
                category='quality',
                message='Answer is empty',
                auto_fixable=True
            ))
            result.quality_score *= 0.0
            return
        
        # Check length
        answer_len = len(answer)
        
        if answer_len < self.min_answer_length:
            result.issues.append(ValidationIssue(
                severity='warning',
                category='quality',
                message=f'Answer too short ({answer_len} chars, min {self.min_answer_length})',
                details={'length': answer_len, 'min_required': self.min_answer_length},
                auto_fixable=True
            ))
            result.quality_score *= 0.8
        
        if answer_len > self.max_answer_length:
            result.warnings.append(f'Answer is long ({answer_len} chars)')
            result.quality_score *= 0.95
        
        # Check if answer is just error message
        error_phrases = ['error', 'failed', 'unable to', 'could not', 'no results']
        answer_lower = answer.lower()
        
        if any(phrase in answer_lower for phrase in error_phrases):
            if len(answer) < 100:
                result.issues.append(ValidationIssue(
                    severity='critical',
                    category='quality',
                    message='Answer appears to be an error message',
                    auto_fixable=True
                ))
                result.quality_score *= 0.5
        
        # Check for completeness (ends with punctuation)
        if answer and not answer.rstrip().endswith(('.', '!', '?', '"', "'")):
            result.issues.append(ValidationIssue(
                severity='warning',
                category='quality',
                message='Answer appears incomplete (no ending punctuation)',
                auto_fixable=True
            ))
            result.quality_score *= 0.9
    
    
    def _validate_citations(self, answer: str, result: ValidationResult):
        """Validate citations in answer"""
        
        # Match multiple citation formats:
        # [doc:1], [source:2], [Sources: file.pdf Page 3], [Document 1, Document 2]
        patterns = [
            r'\[(?:doc|source|ref):\d+\]',                    # [doc:1], [source:2]
            r'\[Sources?:\s*[^\]]+\]',                        # [Sources: file.pdf Pages 2-3]
            r'\[Document\s+\d+(?:,\s*Document\s+\d+)*\]'     # [Document 1, Document 2]
        ]
        
        all_citations = []
        for pattern in patterns:
            citations = re.findall(pattern, answer, re.IGNORECASE)
            all_citations.extend(citations)
        
        # Remove duplicates
        unique_citations = list(set(all_citations))
        
        result.citation_count = len(unique_citations)
        result.has_citations = len(unique_citations) > 0
        
        if len(unique_citations) < self.min_citations:
            result.issues.append(ValidationIssue(
                severity='warning',
                category='citation',
                message=f'Insufficient citations ({len(unique_citations)}, need {self.min_citations})',
                details={
                    'found': len(unique_citations),
                    'required': self.min_citations,
                    'citations': unique_citations[:5]  # Show first 5
                },
                auto_fixable=True
            ))
            result.quality_score *= 0.9
        
        # Check for broken citation format
        broken_patterns = [r'\[doc:\]', r'\[:\d+\]', r'\[doc\d+\]']
        broken_citations = []
        for pattern in broken_patterns:
            broken = re.findall(pattern, answer)
            broken_citations.extend(broken)
        
        if broken_citations:
            result.issues.append(ValidationIssue(
                severity='warning',
                category='citation',
                message=f'Found {len(broken_citations)} malformed citations',
                details={'examples': broken_citations[:3]},
                auto_fixable=True
            ))
            result.quality_score *= 0.95



    # ========================================================================
    # LLM-as-Judge Validation
    # ========================================================================
    
    def _llm_judge_validate(
        self,
        answer: str,
        query: str,
        context_chunks: Optional[List[str]],
        result: ValidationResult
    ):
        """Use LLM to validate answer quality, hallucination, and relevance"""
        
        logger.info("Running LLM-as-Judge validation...")
        
        # Prepare context
        # context_text = "\n\n".join(context_chunks) if context_chunks else "No context provided"
        # Prepare context - handle both list of strings and list of dicts
        if context_chunks:
            # Extract text from chunks (handle both str and dict formats)
            chunk_texts = []
            for chunk in context_chunks:
                if isinstance(chunk, dict):
                    # If chunk is a dict, extract 'text' field
                    chunk_texts.append(chunk.get('text', str(chunk)))
                else:
                    # If chunk is already a string
                    chunk_texts.append(str(chunk))
            context_text = "\n\n".join(chunk_texts)
        else:
            context_text = "No context provided"

        
        # Prompt
        prompt = f"""You are an expert answer validator for a RAG (Retrieval-Augmented Generation) system.

CONTEXT DOCUMENTS:
{context_text[:2000]}

USER QUESTION:
{query}

GENERATED ANSWER:
{answer}

TASK: Validate the answer across multiple dimensions:

1. **Hallucination Check**: Is the answer fully grounded in the context documents? Does it contain information NOT present in the context?

2. **Relevance Check**: Does the answer directly address the user's question?

3. **Quality Check**: Is the answer complete, clear, and well-structured?

Respond in JSON format:
{{
    "is_grounded": true/false,
    "hallucinated_claims": ["claim1", "claim2"],
    "relevance_score": 0.0-1.0,
    "quality_score": 0.0-1.0,
    "confidence": 0.0-1.0,
    "issues": ["issue1", "issue2"],
    "reasoning": "brief explanation"
}}"""
        
        try:
            # Call LLM using deployment name
            response = self.llm_client.chat.completions.create(
                model=self.llm_deployment,
                messages=[
                    {"role": "system", "content": "You are a precise answer validator. Always respond in valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            llm_result = json.loads(response.choices[0].message.content)
            
            # Store reasoning
            result.llm_judge_reasoning = llm_result.get('reasoning', '')
            
            # Process hallucination
            if not llm_result.get('is_grounded', True):
                result.has_hallucination = True
                result.issues.append(ValidationIssue(
                    severity='warning',
                    category='hallucination',
                    message='LLM detected potential hallucination',
                    details={
                        'claims': llm_result.get('hallucinated_claims', []),
                        'reasoning': llm_result.get('reasoning', '')
                    },
                    auto_fixable=True
                ))
                result.confidence_score *= 0.7
            
            # Process relevance
            relevance = llm_result.get('relevance_score', 0.8)
            result.confidence_score *= relevance
            
            if relevance < 0.5:
                result.issues.append(ValidationIssue(
                    severity='warning',
                    category='quality',
                    message=f'Low relevance score: {relevance:.2f}',
                    details={'relevance_score': relevance},
                    auto_fixable=True
                ))
            
            # Process quality
            quality = llm_result.get('quality_score', 0.8)
            result.quality_score *= quality
            
            if quality < 0.6:
                result.issues.append(ValidationIssue(
                    severity='warning',
                    category='quality',
                    message=f'Low quality score: {quality:.2f}',
                    details={'quality_score': quality},
                    auto_fixable=True
                ))
            
            # Add any additional issues
            for issue_text in llm_result.get('issues', []):
                result.warnings.append(f"LLM: {issue_text}")
            
            logger.info(f"✅ LLM judge complete: grounded={llm_result.get('is_grounded')}, "
                       f"relevance={relevance:.2f}, quality={quality:.2f}")
        
        except Exception as e:
            logger.error(f"LLM judge error: {e}")
            raise
    
    
    # ========================================================================
    # Rule-Based Fallback Validation
    # ========================================================================
    
    def _rule_based_validation(
        self,
        answer: str,
        query: str,
        context_chunks: Optional[List[str]],
        result: ValidationResult
    ):
        """Fallback rule-based validation when LLM is not available"""
        
        logger.info("Using rule-based validation (fallback)")
        
        # Hallucination detection (rule-based)
        self._detect_hallucination_rules(answer, query, context_chunks, result)
        
        # Relevance scoring (rule-based)
        self._score_relevance_rules(answer, query, result)
        
        # Quality assessment (rule-based)
        self._assess_quality_rules(answer, result)
    
    
    def _detect_hallucination_rules(
        self,
        answer: str,
        query: str,
        context_chunks: Optional[List[str]],
        result: ValidationResult
    ):
        """Rule-based hallucination detection"""
        
        answer_lower = answer.lower()
        
        # Check for hallucination indicator phrases
        found_phrases = []
        for phrase in self.hallucination_phrases:
            if phrase in answer_lower:
                found_phrases.append(phrase)
        
        if found_phrases:
            result.has_hallucination = True
            result.issues.append(ValidationIssue(
                severity='warning',
                category='hallucination',
                message='Answer contains uncertain language',
                details={'phrases': found_phrases[:3]},
                auto_fixable=True
            ))
            result.confidence_score *= 0.7
        
        # Check grounding in context
        if context_chunks:
            self._check_grounding_rules(answer, context_chunks, result)
    
    
    def _check_grounding_rules(
        self,
        answer: str,
        context_chunks: List[str],
        result: ValidationResult
    ):
        """Check if answer is grounded in provided context (rule-based)"""
        
        # Check if key terms in answer appear in context
        answer_words = set(answer.lower().split())
        # context_text = ' '.join(context_chunks).lower()
        # Extract text from chunks
        chunk_texts = []
        for chunk in context_chunks:
            if isinstance(chunk, dict):
                chunk_texts.append(chunk.get('text', str(chunk)))
            else:
                chunk_texts.append(str(chunk))
        context_text = ' '.join(chunk_texts).lower()

        context_words = set(context_text.split())
        
        # Calculate overlap
        overlap = len(answer_words & context_words) / len(answer_words) if answer_words else 0
        
        if overlap < 0.3:
            result.issues.append(ValidationIssue(
                severity='warning',
                category='hallucination',
                message=f'Low grounding in context (overlap: {overlap:.1%})',
                details={'overlap_ratio': overlap},
                auto_fixable=True
            ))
            result.confidence_score *= 0.85
    
    
    def _score_relevance_rules(self, answer: str, query: str, result: ValidationResult):
        """Rule-based relevance scoring"""
        
        # Keyword-based relevance
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'what',
                     'who', 'when', 'where', 'why', 'how', 'which'}
        
        query_keywords = query_words - stop_words
        answer_keywords = answer_words - stop_words
        
        if not query_keywords:
            return
        
        # Calculate coverage
        coverage = len(query_keywords & answer_keywords) / len(query_keywords)
        
        result.confidence_score *= (0.5 + coverage * 0.5)
        
        if coverage < 0.3:
            result.issues.append(ValidationIssue(
                severity='warning',
                category='quality',
                message=f'Low relevance to query ({coverage:.0%} keyword coverage)',
                details={'coverage': coverage},
                auto_fixable=True
            ))
    
    
    def _assess_quality_rules(self, answer: str, result: ValidationResult):
        """Rule-based quality assessment"""
        
        # Check for weak/uncertain language
        answer_lower = answer.lower()
        weak_count = sum(1 for phrase in self.weak_phrases if phrase in answer_lower)
        
        if weak_count > 3:
            result.warnings.append(f'Answer contains {weak_count} uncertain phrases')
            result.confidence_score *= 0.9
        
        # Check sentence structure
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        
        if len(sentences) == 1 and len(answer) > 200:
            result.warnings.append('Answer is a single long sentence (readability issue)')
            result.quality_score *= 0.95
    
    
    def _check_safety(self, answer: str, result: ValidationResult):
        """Minimal safety checks (no PII checks)"""
        
        answer_lower = answer.lower()
        
        # Only check for truly inappropriate content
        inappropriate_keywords = ['offensive', 'discriminatory', 'hate speech', 'violence']
        if any(kw in answer_lower for kw in inappropriate_keywords):
            result.issues.append(ValidationIssue(
                severity='critical',
                category='safety',
                message='Potentially inappropriate content detected',
                auto_fixable=False
            ))
            result.is_valid = False
    
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    def get_validation_summary(self, result: ValidationResult) -> str:
        """Get human-readable validation summary"""
        
        lines = [
            f"📊 Validation Summary",
            f"   Status: {'✅ VALID' if result.is_valid else '❌ INVALID'}",
            f"   Quality Score: {result.quality_score:.2f}/1.0",
            f"   Confidence: {result.confidence_score:.2f}/1.0",
            f"   Answer Length: {result.answer_length} chars",
            f"   Citations: {result.citation_count}",
            f"   Hallucination: {'⚠️ Yes' if result.has_hallucination else '✅ No'}",
            f"   LLM Judge: {'✅ Used' if result.llm_judge_used else '❌ Not used'}",
        ]
        
        if result.llm_judge_reasoning:
            lines.append(f"\n🤖 LLM Judge Reasoning:")
            lines.append(f"   {result.llm_judge_reasoning[:200]}")
        
        if result.issues:
            lines.append(f"\n⚠️  Issues Found: {len(result.issues)}")
            for issue in result.issues[:5]:
                lines.append(f"   - [{issue.severity.upper()}] {issue.category}: {issue.message}")
        
        if result.warnings:
            lines.append(f"\n💡 Warnings: {len(result.warnings)}")
            for warning in result.warnings[:3]:
                lines.append(f"   - {warning}")
        
        return "\n".join(lines)


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_answer(answer: str, query: str, **kwargs) -> ValidationResult:
    """
    Quick validation function
    
    Args:
        answer: Generated answer
        query: User query
        **kwargs: Additional ValidationTools parameters
        
    Returns:
        ValidationResult
    """
    validator = ValidationTools(**kwargs)
    return validator.validate_answer(answer, query)

# ============================================================================
# CrewAI Tool Wrappers
# ============================================================================

@tool("Validate Answer")
def validate_answer_tool(
    answer: str,
    query: str,
    context_chunks: list = None,
    metadata: dict = None
) -> dict:
    """
    Validate generated answer for quality, hallucinations, and citations.
    
    Uses LLM-as-Judge for comprehensive validation.
    
    Args:
        answer: Generated answer to validate
        query: Original user query
        context_chunks: Retrieved context chunks used
        metadata: Additional metadata
        
    Returns:
        dict: ValidationResult as dict with is_valid, quality_score, 
              confidence_score, issues, citations, hallucination status
    """
    validator = ValidationTools(
        enable_llm_judge=True,
        min_quality_score=0.6,
        min_citations=1
    )
    result = validator.validate_answer(
        answer=answer,
        query=query,
        context_chunks=context_chunks,
        metadata=metadata
    )
    return result.to_dict()


@tool("Get Validation Summary")
def get_validation_summary_tool(validation_result: dict) -> str:
    """
    Get human-readable validation summary.
    
    Args:
        validation_result: ValidationResult dict from validate_answer_tool
        
    Returns:
        str: Formatted validation summary
    """
    validator = ValidationTools()
    # Reconstruct ValidationResult from dict
    result = ValidationResult(
        is_valid=validation_result['is_valid'],
        quality_score=validation_result['quality_score'],
        confidence_score=validation_result['confidence_score'],
        has_hallucination=validation_result['has_hallucination'],
        citation_count=validation_result['citation_count']
    )
    return validator.get_validation_summary(result)

