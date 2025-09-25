# -*- coding: utf-8 -*-
"""
Knowledge Graph Pipeline for PDF Document Processing
Extracts entities, disambiguates them with Google Knowledge Graph API,
and refines with a second LLM chain for entity grounding before storing in Neo4j.
Entities are stored without Document nodes. Multiple labels are supported:
- If KG returns multiple labels, remove "Thing" and keep the rest.
- If the only label is "Thing", let the LLM infer a more meaningful label.
Entity names are normalized to Title Case.
Entities can also store multiple dynamic properties from KG and additional properties inferred by LLM from chunk text.
Entity labels are normalized to avoid Neo4j Cypher syntax errors.
Relationships are also refined by the LLM to ensure meaningful links (e.g., City in Country, Person bornIn City).
"""

import asyncio
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

import requests
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper()),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        # LLM Provider Configuration
        self.llm_provider = os.getenv('LLM_PROVIDER', 'openai').lower()  # 'openai' or 'anthropic'
        
        # API Keys
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.google_kg_api_key = self._get_required_env('GOOGLE_KG_API_KEY')
        
        # Neo4j Configuration
        self.neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = self._get_required_env('NEO4J_PASSWORD')
        
        # LLM Model Configuration
        if self.llm_provider == 'openai':
            self.llm_model = os.getenv('OPENAI_MODEL', 'gpt-4')
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is required when using OpenAI provider")
        elif self.llm_provider == 'anthropic':
            self.llm_model = os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022')
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required when using Anthropic provider")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}. Use 'openai' or 'anthropic'")
        
        # LLM Parameters
        self.llm_temperature = float(os.getenv('LLM_TEMPERATURE', '0.0'))
        self.max_tokens = int(os.getenv('MAX_TOKENS', '4000'))
        
        # Document Processing Configuration
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
        
        # Knowledge Graph Configuration
        self.kg_search_limit = int(os.getenv('KG_SEARCH_LIMIT', '3'))
        self.kg_languages = os.getenv('KG_LANGUAGES', 'en').split(',')
        
        # Network Configuration
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '30'))
        
        self._validate_config()
    
    def _get_required_env(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        return value

    def _validate_config(self):
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be less than CHUNK_SIZE")
        if not (0 <= self.llm_temperature <= 2):
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2")
        if self.max_tokens <= 0:
            raise ValueError("MAX_TOKENS must be positive")
        
        logger.info(f"Using LLM provider: {self.llm_provider}")
        logger.info(f"Using model: {self.llm_model}")
        logger.info(f"Temperature: {self.llm_temperature}")
        logger.info(f"Max tokens: {self.max_tokens}")

class Entity(BaseModel):
    name: str
    entity_type: str
    context: str
    confidence: float

class Relationship(BaseModel):
    source_entity: str
    target_entity: str
    relationship_type: str
    context: str
    confidence: float

class EntityExtraction(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]
    summary: str

class KGProperty(BaseModel):
    key: str
    value: str

class KnowledgeGraphEntity(BaseModel):
    original_name: str
    kg_name: Optional[str] = None
    kg_id: Optional[str] = None
    kg_types: List[str] = Field(default_factory=list)
    kg_description: Optional[str] = None
    kg_score: Optional[float] = None
    context: str
    confidence: float
    properties: List[KGProperty] = Field(default_factory=list)

class KnowledgeGraphRelationship(BaseModel):
    source_entity: str
    target_entity: str
    relationship_type: str
    context: str
    confidence: float
    kg_source_name: Optional[str] = None
    kg_target_name: Optional[str] = None

class RefinedEntities(BaseModel):
    entities: List[KnowledgeGraphEntity]

class RefinedRelationships(BaseModel):
    relationships: List[KnowledgeGraphRelationship]

@dataclass
class ProcessingContext:
    document_id: str
    chunk_summaries: List[str]
    processed_entities: List[KnowledgeGraphEntity]
    processed_relationships: List[KnowledgeGraphRelationship]
    overall_summary: str = ""

class KnowledgeGraphSearcher:
    def __init__(self, api_key: str, config: Config = None):
        self.api_key = api_key
        self.config = config or Config()
        self.base_url = "https://kgsearch.googleapis.com/v1/entities:search"
        self.cache = {}

    async def search_entity(self, query: str, limit: int = None, languages: List[str] = None) -> Dict:
        limit = limit or self.config.kg_search_limit
        languages = languages or self.config.kg_languages
        cache_key = f"{query.lower()}_{limit}_{languages}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        params = {'query': query, 'key': self.api_key, 'limit': limit, 'indent': True}
        if languages:
            params['languages'] = ','.join(languages)
        try:
            logger.debug(f"Searching KG for entity: {query}")
            response = requests.get(self.base_url, params=params, timeout=self.config.request_timeout)
            response.raise_for_status()
            result = response.json()
            self.cache[cache_key] = result
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching entity '{query}': {e}")
            return {}

class Neo4jConnector:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def normalize_name(self, name: str) -> str:
        """Normalize entity name to Title Case, handling KG IDs properly and preserving Unicode"""
        if name.startswith("Kg:/") or name.startswith("kg:/"):
            # If it's a KG ID format, this shouldn't be used as display name
            logger.warning(f"Attempting to use KG ID as name: {name}")
            return name
        
        # Preserve Unicode characters and handle capitalization properly
        if not name:
            return name
            
        # Split by spaces and capitalize each word while preserving Unicode
        words = name.split()
        normalized_words = []
        
        for word in words:
            if word:
                # Capitalize first letter while preserving Unicode characters
                normalized_word = word[0].upper() + word[1:].lower() if len(word) > 1 else word.upper()
                normalized_words.append(normalized_word)
        
        result = " ".join(normalized_words)
        logger.debug(f"Normalized '{name}' to '{result}'")
        return result
    
    def normalize_labels(self, kg_types: List[str]) -> str:
        if not kg_types:
            # If no types provided, this entity should have been filtered out earlier
            # or the LLM should have provided a meaningful type
            logger.warning("Entity has no types - this should not happen with proper LLM labeling")
            return "Unknown"  # Temporary fallback that signals an issue
        
        # Clean schema.org URLs and filter out "Thing"
        cleaned = [t.replace("http://schema.org/", "") for t in kg_types if t != "Thing"]
        
        if not cleaned:
            # If only "Thing" was provided, this should have been handled by the LLM
            logger.warning("Entity only has 'Thing' type - LLM should have inferred a better label")
            return "Unknown"  # Temporary fallback that signals an issue
        
        # Make labels safe for Neo4j
        safe_labels = [re.sub(r"[^A-Za-z0-9]", "", lbl.title()) for lbl in cleaned]
        return ":".join(safe_labels)
    
    def create_entity(self, entity: KnowledgeGraphEntity, document_id: str):
        """Create entity using the proper name, not KG ID"""
        # Use kg_name if available, otherwise use original_name
        display_name = entity.kg_name or entity.original_name
        norm_name = self.normalize_name(display_name)
        labels = self.normalize_labels(entity.kg_types)
        
        logger.info(f"Storing entity: {norm_name} ({labels}) [KG ID: {entity.kg_id}]")
        
        # Store both normalized name and original for debugging
        self.entity_name_mapping = getattr(self, 'entity_name_mapping', {})
        self.entity_name_mapping[entity.original_name] = norm_name
        if entity.kg_name:
            self.entity_name_mapping[entity.kg_name] = norm_name
        
        with self.driver.session() as session:
            try:
                session.execute_write(self._create_entity_tx, entity, norm_name, labels)
                logger.debug(f"Successfully stored entity: {norm_name}")
            except Exception as e:
                logger.error(f"Error storing entity {norm_name}: {e}")
                raise
    
    def create_relationship(self, relationship: KnowledgeGraphRelationship, document_id: str):
        """Create relationship using proper entity names, not KG IDs"""
        # Use the entity name mapping to get the actual stored names
        entity_mapping = getattr(self, 'entity_name_mapping', {})
        
        # Use kg names if available, otherwise use original names
        source_name = relationship.kg_source_name or relationship.source_entity
        target_name = relationship.kg_target_name or relationship.target_entity
        
        # Try to find the normalized names from our mapping
        norm_source = entity_mapping.get(source_name, self.normalize_name(source_name))
        norm_target = entity_mapping.get(target_name, self.normalize_name(target_name))
        
        logger.info(f"Storing relationship: {norm_source} -[{relationship.relationship_type}]-> {norm_target}")
        logger.debug(f"Original names: {source_name} -> {target_name}")
        logger.debug(f"Normalized names: {norm_source} -> {norm_target}")
        
        with self.driver.session() as session:
            try:
                result = session.execute_write(
                    self._create_relationship_tx, 
                    relationship, 
                    norm_source, 
                    norm_target
                )
                if result and result.get('relationships_created', 0) > 0:
                    logger.debug(f"Successfully created relationship: {norm_source} -> {norm_target}")
                else:
                    # Try to debug what entities actually exist
                    existing_entities = session.run("MATCH (n) RETURN n.name LIMIT 20").values()
                    logger.warning(f"Available entities: {[e[0] for e in existing_entities]}")
                    logger.warning(f"No relationship created between {norm_source} and {norm_target} - entities may not exist")
            except Exception as e:
                logger.error(f"Error creating relationship {norm_source} -> {norm_target}: {e}")
                raise
    
    @staticmethod
    def _create_entity_tx(tx, entity: KnowledgeGraphEntity, norm_name: str, labels: str):
        # Convert properties to dict
        props = {prop.key: prop.value for prop in entity.properties}
        
        query = f"""
        MERGE (e:{labels} {{name: $name}})
        ON CREATE SET e.kg_id = $kg_id,
                      e.description = $description,
                      e.created_at = datetime(),
                      e.confidence = $confidence,
                      e.original_name = $original_name
        ON MATCH SET e.last_seen = datetime()
        SET e += $properties
        RETURN e
        """
        
        result = tx.run(
            query,
            name=norm_name,
            kg_id=entity.kg_id,
            description=entity.kg_description,
            confidence=entity.confidence,
            original_name=entity.original_name,
            properties=props
        )
        return result.single()
    
    @staticmethod
    def _create_relationship_tx(tx, relationship: KnowledgeGraphRelationship, source_name: str, target_name: str):
        # Make relationship type safe for Neo4j
        safe_rel_type = re.sub(r"[^A-Za-z0-9]", "_", relationship.relationship_type.upper())
        
        # First verify both entities exist
        verify_query = """
        MATCH (source {name: $source_name})
        MATCH (target {name: $target_name})
        RETURN count(*) as entity_count
        """
        
        verify_result = tx.run(verify_query, source_name=source_name, target_name=target_name)
        entity_count = verify_result.single()["entity_count"]
        
        if entity_count == 0:
            logger.warning(f"Cannot create relationship: entities '{source_name}' and/or '{target_name}' not found")
            return {"relationships_created": 0}
        
        # Create the relationship
        query = f"""
        MATCH (source {{name: $source_name}})
        MATCH (target {{name: $target_name}})
        MERGE (source)-[r:{safe_rel_type}]->(target)
        ON CREATE SET r.context = $context, 
                      r.confidence = $confidence, 
                      r.created_at = datetime()
        ON MATCH SET r.last_seen = datetime(), 
                     r.frequency = coalesce(r.frequency, 0) + 1
        RETURN count(r) as relationships_created
        """
        
        result = tx.run(
            query, 
            source_name=source_name, 
            target_name=target_name,
            context=relationship.context, 
            confidence=relationship.confidence
        )
        
        return result.single()

class KnowledgeGraphPipeline:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.llm = self._initialize_llm()
        self.kg_searcher = KnowledgeGraphSearcher(self.config.google_kg_api_key, self.config)
        self.neo4j = Neo4jConnector(self.config.neo4j_uri, self.config.neo4j_user, self.config.neo4j_password)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size, 
            chunk_overlap=self.config.chunk_overlap
        )
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on the provider configuration"""
        if self.config.llm_provider == 'openai':
            logger.info(f"Initializing OpenAI LLM with model: {self.config.llm_model}")
            return ChatOpenAI(
                api_key=self.config.openai_api_key,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.max_tokens
            )
        elif self.config.llm_provider == 'anthropic':
            logger.info(f"Initializing Anthropic LLM with model: {self.config.llm_model}")
            return ChatAnthropic(
                api_key=self.config.anthropic_api_key,
                model=self.config.llm_model,
                temperature=self.config.llm_temperature,
                max_tokens=self.config.max_tokens
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm_provider}")
    
    def get_llm_info(self) -> dict:
        """Get information about the currently configured LLM"""
        return {
            "provider": self.config.llm_provider,
            "model": self.config.llm_model,
            "temperature": self.config.llm_temperature,
            "max_tokens": self.config.max_tokens
        }

    async def load_pdf(self, file_path: str) -> List[Any]:
        logger.info(f"Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = []
        async for page in loader.alazy_load():
            pages.append(page)
        return pages

    async def extract_entities_from_text(self, text: str, context: ProcessingContext) -> EntityExtraction:
        logger.debug("Extracting entities and relationships with LLM")
        system_prompt = """You are an expert entity and relationship extraction system. 
        Extract ONLY the most important and relevant named entities that are central to the content.
        Focus on:
        - Key people (names, not generic roles)
        - Important places (cities, countries, institutions)
        - Significant organizations, companies, universities
        - Notable concepts, theories, discoveries, or achievements
        - Important events (but NOT standalone dates)
        
        AVOID extracting:
        - Generic terms, common nouns, or adjectives
        - Overly broad concepts
        - Minor details or passing mentions
        - Standalone dates, years, or time periods (like "1891", "November 7, 1867")
        - Generic date references (like "birth date", "death date")
        
        For entity_type, use specific, meaningful categories like:
        - Person, Scientist, Physicist, Chemist
        - City, Country, University, Laboratory
        - Discovery, Theory, Award, Element, Technique
        - Field, Research, Institution, Event
        
        DO NOT create entities for:
        - Dates, years, birth dates, death dates
        - Time periods or temporal references
        - Generic temporal concepts
        
        Focus on substantive entities that represent real-world things, people, places, and concepts.
        Provide confidence scores between 0.0 and 1.0 based on importance and specificity."""
        
        structured_llm = self.llm.with_structured_output(EntityExtraction)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=text)]
        return await structured_llm.ainvoke(messages)

    async def refine_with_kg(self, chunk: str, entities: List[Entity], kg_candidates: Dict[str, List[Dict]]) -> List[KnowledgeGraphEntity]:
        logger.debug("Refining entities with KG results via LLM")
        
        kg_info = []
        for ent in entities:
            matches = kg_candidates.get(ent.name, [])
            formatted = []
            for m in matches:
                result = m.get("result", {})
                
                # Extract image URL if available
                image_url = None
                if "image" in result:
                    if isinstance(result["image"], dict) and "contentUrl" in result["image"]:
                        image_url = result["image"]["contentUrl"]
                    elif isinstance(result["image"], str):
                        image_url = result["image"]
                
                # Collect all extra properties
                extra_props = {k: v for k, v in result.items() 
                             if k not in ["name", "@id", "@type", "description", "image"]}
                
                # Add image URL to extra properties if found
                if image_url:
                    extra_props["image_url"] = image_url
                
                formatted.append({
                    "name": result.get("name"),
                    "id": result.get("@id"),
                    "types": result.get("@type"),
                    "description": result.get("description"),
                    "score": m.get("resultScore"),
                    "image_url": image_url,
                    "extra": extra_props
                })
            kg_info.append({"entity": ent.name, "candidates": formatted})

        system_prompt = """
You are an expert entity disambiguator and labeler.
Given extracted entities, their context, and Google Knowledge Graph search results:

FOR ENTITY SELECTION:
- Select the best KG match ONLY if you're confident it's the same entity
- Be conservative - if uncertain, keep the original entity without KG data
- ALWAYS use the human-readable name from KG results, never the ID

FOR ENTITY LABELING - CRITICAL:
- EVERY entity MUST have meaningful, specific labels
- NEVER leave kg_types empty or use generic labels like "Thing" or "Entity"
- DO NOT create entities for dates, years, or temporal references
- If KG provides good labels, use them (excluding "Thing")
- If KG only has "Thing" or no KG match, you MUST infer specific labels from context:
  * For people: Person, Scientist, Physicist, Chemist, Researcher, Nobel_Laureate, etc.
  * For places: City, Country, Region, University, Laboratory, Institute, etc. 
  * For organizations: University, Company, Institution, Laboratory, Research_Center, etc.
  * For concepts: Theory, Discovery, Award, Element, Method, Technique, Field_Of_Study, etc.
  * For achievements: Nobel_Prize, Discovery, Publication, Patent, Award, etc.
- Use SPECIFIC labels that describe what the entity actually is
- Multiple labels are encouraged (e.g., ["Person", "Scientist", "Physicist"])

FORBIDDEN ENTITY TYPES:
- DO NOT create entities with types like: Date, Birth_Date, Death_Date, Year, Time, Period, Temporal
- DO NOT create entities for standalone dates, years, or time references
- Filter out any date-related entities from your results

FOR PROPERTIES:
- Include useful properties from both KG and context
- Examples: birth_date, death_date, nationality, field_of_study, discovery_year, etc.
- Dates should be stored as PROPERTIES of other entities, not as separate entities
- ALWAYS include image_url as a property if available from KG results

FOR IMAGES:
- When KG results contain image_url, ALWAYS include it as a property
- Store the image URL exactly as provided
- This helps visualize entities in the knowledge graph

QUALITY CONTROL:
- Only return entities that are truly important to the document
- Filter out overly generic or minor entities
- Filter out ALL date-related entities
- Ensure entity names are clean and properly formatted
- MANDATORY: Every entity must have at least one specific, meaningful label

EXAMPLES OF GOOD ENTITIES:
- Marie Curie → kg_types: ["Person", "Scientist", "Physicist", "Nobel_Laureate"], properties: [{"key": "image_url", "value": "https://..."}]
- University of Paris → kg_types: ["University", "Educational_Institution"]
- Radium → kg_types: ["Chemical_Element", "Discovery"]

EXAMPLES OF FORBIDDEN ENTITIES:
- "November 7, 1867" → DO NOT CREATE
- "1891" → DO NOT CREATE  
- "birth date" → DO NOT CREATE
"""
        
        structured_llm = self.llm.with_structured_output(RefinedEntities)
        messages = [
            SystemMessage(content=system_prompt), 
            HumanMessage(content=json.dumps({
                "chunk": chunk, 
                "entities": [e.model_dump() for e in entities], 
                "kg_results": kg_info
            }, indent=2))
        ]
        
        refined = await structured_llm.ainvoke(messages)
        return refined.entities

    async def refine_relationships(self, chunk: str, entities: List[KnowledgeGraphEntity], relationships: List[Relationship]) -> List[KnowledgeGraphRelationship]:
        logger.debug("Refining relationships with LLM")
        
        # Create a clear mapping of available entity names for the LLM
        entity_name_list = []
        for entity in entities:
            display_name = entity.kg_name or entity.original_name
            entity_name_list.append({
                "name": display_name,
                "original_name": entity.original_name,
                "kg_name": entity.kg_name,
                "kg_id": entity.kg_id,
                "types": entity.kg_types
            })
        
        system_prompt = """
You are an expert in knowledge graphs and relationship modeling.
Given enriched entities with their types and properties, generate meaningful relationships.

CRITICAL REQUIREMENTS:
1. ONLY use the "name" field from the entity list below for source_entity and target_entity
2. NEVER use kg_id values (like kg:/m/053_d) in relationships
3. Use the exact "name" values as they appear in the entity list
4. Both source_entity and target_entity must exist in the provided entity list

RELATIONSHIP GUIDELINES:
- Use domain-appropriate predicates:
  * Person relationships: bornIn, diedIn, marriedTo, childOf, siblingOf, studiedAt, workedAt, discoveredBy, wonAward
  * Location relationships: locatedIn, partOf, capitalOf, nearTo
  * Organization relationships: foundedBy, basedIn, affiliatedWith, memberOf
  * Academic/Scientific: studiedAt, researchedAt, discoveredBy, publishedBy, awardedBy, supervisedBy
  * Temporal: occurredIn, happenedAt, establishedIn, bornIn (with dates)

QUALITY CONTROL:
- Each relationship should be factually supported by the text
- Avoid redundant or overly obvious relationships
- Prefer more specific relationship types over generic ones
- Double-check that both entity names exactly match the "name" field in the entity list

EXAMPLE:
If entity list contains: {"name": "Maria Skłodowska-curie", "kg_id": "kg:/m/053_d"}
Then use: "Maria Skłodowska-curie" NOT "kg:/m/053_d"
"""
        
        structured_llm = self.llm.with_structured_output(RefinedRelationships)
        messages = [
            SystemMessage(content=system_prompt), 
            HumanMessage(content=json.dumps({
                "chunk": chunk, 
                "available_entities": entity_name_list,
                "extracted_relationships": [r.model_dump() for r in relationships]
            }, indent=2))
        ]
        
        refined = await structured_llm.ainvoke(messages)
        
        # Additional validation to catch any KG IDs that slipped through
        validated_relationships = []
        for rel in refined.relationships:
            if rel.source_entity.startswith(('kg:/', 'Kg:/')):
                logger.warning(f"Filtering out relationship with KG ID source: {rel.source_entity}")
                continue
            if rel.target_entity.startswith(('kg:/', 'Kg:/')):
                logger.warning(f"Filtering out relationship with KG ID target: {rel.target_entity}")
                continue
            validated_relationships.append(rel)
        
        logger.debug(f"Refined {len(relationships)} relationships into {len(validated_relationships)} valid relationships")
        return validated_relationships

    async def process_document_chunk(self, chunk: str, context: ProcessingContext) -> Tuple[List[KnowledgeGraphEntity], List[KnowledgeGraphRelationship]]:
        logger.info(f"Processing chunk for document {context.document_id}")
        
        # Extract entities and relationships
        extraction = await self.extract_entities_from_text(chunk, context)
        context.chunk_summaries.append(extraction.summary)
        
        # Search KG for each entity
        kg_candidates = {}
        for entity in extraction.entities:
            results = await self.kg_searcher.search_entity(entity.name)
            kg_candidates[entity.name] = results.get("itemListElement", [])
        
        # Refine entities with KG information
        enriched_entities = await self.refine_with_kg(chunk, extraction.entities, kg_candidates)
        
        # Refine relationships
        refined_relationships = await self.refine_relationships(chunk, enriched_entities, extraction.relationships)
        
        return enriched_entities, refined_relationships

    async def generate_overall_summary(self, context: ProcessingContext) -> str:
        if not context.chunk_summaries:
            return ""
        
        logger.info("Generating overall summary")
        messages = [
            SystemMessage(content="Summarize the key themes and topics from this document based on the chunk summaries."), 
            HumanMessage(content="\n".join(context.chunk_summaries))
        ]
        response = await self.llm.ainvoke(messages)
        return response.content

    async def process_pdf_document(self, file_path: str, document_id: str = None) -> ProcessingContext:
        if not document_id:
            document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        context = ProcessingContext(document_id, [], [], [])
        logger.info(f"Starting document processing: {file_path}")
        
        # Load and split document
        pages = await self.load_pdf(file_path)
        full_text = "\n\n".join([p.page_content for p in pages])
        chunks = self.text_splitter.split_text(full_text)
        
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Process each chunk
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}")
            
            chunk_entities, chunk_relationships = await self.process_document_chunk(chunk, context)
            
            # Store entities first (with enhanced filtering)
            for entity in chunk_entities:
                # Skip entities with very low confidence
                if entity.confidence < 0.3:
                    logger.debug(f"Skipping low-confidence entity: {entity.original_name} (confidence: {entity.confidence})")
                    continue
                    
                # Skip overly generic entity names
                generic_terms = {'thing', 'entity', 'item', 'concept', 'idea', 'topic', 'subject', 'matter', 'unknown'}
                if entity.original_name.lower().strip() in generic_terms:
                    logger.debug(f"Skipping generic entity: {entity.original_name}")
                    continue
                
                # Skip date-related entities
                date_related_types = {'date', 'birth_date', 'death_date', 'year', 'time', 'period', 'temporal', 'discovery_date', 'award_date'}
                entity_types_lower = [t.lower().replace('_', '').replace('-', '') for t in entity.kg_types]
                if any(date_type.replace('_', '').replace('-', '') in entity_types_lower for date_type in date_related_types):
                    logger.debug(f"Skipping date-related entity: {entity.original_name} (types: {entity.kg_types})")
                    continue
                
                # Skip entities that look like dates or years
                entity_name_clean = entity.original_name.strip()
                if (entity_name_clean.isdigit() and len(entity_name_clean) == 4) or \
                   any(month in entity_name_clean.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 
                                                                        'july', 'august', 'september', 'october', 'november', 'december']):
                    logger.debug(f"Skipping date-like entity: {entity.original_name}")
                    continue
                
                # Skip entities with no meaningful labels
                if not entity.kg_types or all(t.lower() in ['thing', 'entity', 'unknown', ''] for t in entity.kg_types):
                    logger.warning(f"Skipping entity with no meaningful labels: {entity.original_name} (types: {entity.kg_types})")
                    continue
                
                # Log what we're actually storing
                display_name = entity.kg_name or entity.original_name
                has_image = any(prop.key == 'image_url' for prop in entity.properties)
                logger.debug(f"Storing entity: {display_name} with types: {entity.kg_types} (has_image: {has_image})")
                
                self.neo4j.create_entity(entity, document_id)
                context.processed_entities.append(entity)
            
            # Then store relationships
            for relationship in chunk_relationships:
                self.neo4j.create_relationship(relationship, document_id)
                context.processed_relationships.append(relationship)
        
        # Generate overall summary
        context.overall_summary = await self.generate_overall_summary(context)
        
        logger.info(f"Processing complete: {len(context.processed_entities)} entities, {len(context.processed_relationships)} relationships")
        return context

    def close(self):
        self.neo4j.close()

async def main():
    pipeline = None
    try:
        config = Config()
        pipeline = KnowledgeGraphPipeline(config)
        
        # Log LLM information
        llm_info = pipeline.get_llm_info()
        logger.info(f"Pipeline initialized with {llm_info['provider']} ({llm_info['model']})")
        
        pdf_path = os.getenv('PDF_PATH', 'path/to/document.pdf')
        context = await pipeline.process_pdf_document(pdf_path)
        
        print(f"Processed {len(context.processed_entities)} entities")
        print(f"Processed {len(context.processed_relationships)} relationships")
        print(f"Summary: {context.overall_summary}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise
    finally:
        if pipeline:
            pipeline.close()

if __name__ == "__main__":
    asyncio.run(main())