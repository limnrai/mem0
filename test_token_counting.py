#!/usr/bin/env python3
"""
Test script to verify input token counting in Memory.add() function.
Mocks external dependencies to run without API keys.
"""

import os
from unittest.mock import MagicMock, patch
from mem0 import Memory
from mem0.configs.base import MemoryConfig

def test_token_counting():
    """Test that token counting is working in the add() function."""
    
    print("Testing input token counting in Memory.add()...")
    print("-" * 60)
    
    # Mock dependencies to avoid real API calls and credential checks
    with patch('mem0.memory.main.EmbedderFactory') as mock_embedder_factory, \
         patch('mem0.memory.main.VectorStoreFactory') as mock_vector_factory, \
         patch('mem0.memory.main.LlmFactory') as mock_llm_factory, \
         patch('mem0.memory.main.SQLiteManager') as mock_sqlite, \
         patch('mem0.memory.main.capture_event'):
        
        # Setup mocks
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.1] * 1536
        mock_embedder_factory.create.return_value = mock_embedder
        
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = []
        mock_vector_store.config.collection_name = "test_collection"
        mock_vector_factory.create.return_value = mock_vector_store
        
        mock_llm = MagicMock()
        # Mock response for fact extraction (first call)
        # and memory update (second call)
        mock_llm.generate_response.side_effect = [
            '{"facts": ["fact1"]}',  # Fact extraction response
            '{"memory": [{"id": "1", "text": "updated memory", "event": "ADD", "category": "test", "effective_timestamp": "2024-01-01"}]}' # Memory update response
        ]
        mock_llm_factory.create.return_value = mock_llm
        
        # Create Memory instance
        config = MemoryConfig()
        m = Memory(config)
        
        # Test 1: Simple message
        print("\nTest 1: Adding a simple message")
        result = m.add(
            messages="My name is John and I live in New York",
            user_id="test_user"
        )
        
        print(f"Result keys: {result.keys()}")
        assert "token_counts" in result, "token_counts not in result!"
        assert "total_input_tokens" in result["token_counts"], "total_input_tokens not in token_counts!"
        
        token_counts = result["token_counts"]
        print(f"Token counts: {token_counts}")
        print(f"  - Fact extraction tokens: {token_counts['fact_extraction_tokens']}")
        print(f"  - Memory update tokens: {token_counts['memory_update_tokens']}")
        print(f"  - Total input tokens: {token_counts['total_input_tokens']}")
        
        # We expect > 0 because we're mocking the LLM calls but the token counting 
        # uses tiktoken on the input messages which are real strings.
        assert token_counts["total_input_tokens"] > 0, "Total tokens should be greater than 0!"
        print("✓ Test 1 passed!")
        
        # Reset mocks for next test
        mock_llm.generate_response.side_effect = [
            '{"facts": ["fact1"]}', 
            '{"memory": [{"id": "2", "text": "updated memory", "event": "ADD"}]}'
        ]
        
        # Test 2: Conversation with multiple messages
        print("\nTest 2: Adding a conversation")
        result = m.add(
            messages=[
                {"role": "user", "content": "What's the weather like today?"},
                {"role": "assistant", "content": "I don't have access to real-time weather data."}
            ],
            user_id="test_user"
        )
        
        token_counts = result["token_counts"]
        print(f"Token counts: {token_counts}")
        
        assert token_counts["total_input_tokens"] > 0, "Total tokens should be greater than 0!"
        print("✓ Test 2 passed!")
        
        # Test 3: With infer=False (should return 0 tokens)
        print("\nTest 3: Adding with infer=False")
        result = m.add(
            messages="This is a test message",
            user_id="test_user",
            infer=False
        )
        
        token_counts = result["token_counts"]
        print(f"Token counts: {token_counts}")
        assert token_counts["total_input_tokens"] == 0, "Tokens should be 0 when infer=False!"
        print("✓ Test 3 passed!")
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)

if __name__ == "__main__":
    test_token_counting()
