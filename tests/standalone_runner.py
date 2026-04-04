"""Standalone test runner using unittest to bypass pytest environment issues."""

import sys
import os
import unittest
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from ads_ai.agents.video import VideoGenerationAgent
from ads_ai.agents.creative import CreativeAgent
from ads_ai.agents.models import AdScript, CreativeVariants

class TestAgentLogicStandalone(unittest.TestCase):
    """Core logic tests for agents without pytest dependencies."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.mock_client.models = MagicMock()

    def test_video_agent_prompt_synthesis(self):
        """Verify VideoGenerationAgent synthesizes cinematic prompts."""
        agent = VideoGenerationAgent(self.mock_client)
        script = MagicMock(spec=AdScript)
        script.model_dump_json.return_value = '{}'
        plan = MagicMock()
        plan.model_dump_json.return_value = '{}'

        mock_response = MagicMock()
        mock_response.text = "Cinematic 4k render with atmospheric lighting and macro detail."
        self.mock_client.models.generate_content.return_value = mock_response

        prompt = agent._synthesize_video_prompt(script, plan)
        self.assertIn("cinematic", prompt.lower())
        self.assertIn("4k", prompt.lower())
        print("VideoGenerationAgent synthesis: OK")

    def test_creative_agent_variant_generation(self):
        """Verify CreativeAgent produces structured variants."""
        agent = CreativeAgent(self.mock_client)
        mock_variant = AdScript(
            concept_title="T", core_idea="I", hook="H",
            script_scenes=[], brand_integration="B",
            cta="C", video_prompt="V", variant_name="V"
        )
        mock_response = MagicMock()
        mock_response.text = CreativeVariants(variants=[mock_variant]).model_dump_json()
        self.mock_client.models.generate_content.return_value = mock_response

        # Just verify it doesn't crash on empty schemas
        result = agent.generate_variants("P", MagicMock(), MagicMock(), [])
        self.assertEqual(len(result.variants), 1)
        print("CreativeAgent variant generation: OK")

if __name__ == "__main__":
    print("Running Standalone Agent Quality Tests...\n")
    unittest.main()
