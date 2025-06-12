# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=protected-access,unused-variable

import unittest
from unittest.mock import MagicMock, patch

from hugegraph_llm.models.llms.base import BaseLLM
from hugegraph_llm.operators.llm_op.keyword_extract import KeywordExtract


class TestKeywordExtract(unittest.TestCase):
    def setUp(self):
        # Create mock LLM
        self.mock_llm = MagicMock(spec=BaseLLM)
        self.mock_llm.generate.return_value = (
            "KEYWORDS: artificial intelligence, machine learning, neural networks"
        )

        # Sample query
        self.query = (
            "What are the latest advancements in artificial intelligence and machine learning?"
        )

        # Create KeywordExtract instance
        self.extractor = KeywordExtract(
            text=self.query, llm=self.mock_llm, max_keywords=5, language="english"
        )

    def test_init_with_parameters(self):
        """Test initialization with provided parameters."""
        self.assertEqual(self.extractor._query, self.query)
        self.assertEqual(self.extractor._llm, self.mock_llm)
        self.assertEqual(self.extractor._max_keywords, 5)
        self.assertEqual(self.extractor._language, "english")
        self.assertIsNotNone(self.extractor._extract_template)

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        extractor = KeywordExtract()
        self.assertIsNone(extractor._query)
        self.assertIsNone(extractor._llm)
        self.assertEqual(extractor._max_keywords, 5)
        self.assertEqual(extractor._language, "english")
        self.assertIsNotNone(extractor._extract_template)

    def test_init_with_custom_template(self):
        """Test initialization with custom template."""
        custom_template = "Extract keywords from: {question}\nMax keywords: {max_keywords}"
        extractor = KeywordExtract(extract_template=custom_template)
        self.assertEqual(extractor._extract_template, custom_template)

    @patch("hugegraph_llm.operators.llm_op.keyword_extract.LLMs")
    def test_run_with_provided_llm(self, mock_llms_class):
        """Test run method with provided LLM."""
        # Create context
        context = {}

        # Call the method
        result = self.extractor.run(context)

        # Verify that LLMs().get_extract_llm() was not called
        mock_llms_class.assert_not_called()

        # Verify that llm.generate was called
        self.mock_llm.generate.assert_called_once()

        # Verify the result
        self.assertIn("keywords", result)
        self.assertTrue(any("artificial intelligence" in kw for kw in result["keywords"]))
        self.assertTrue(any("machine learning" in kw for kw in result["keywords"]))
        self.assertTrue(any("neural networks" in kw for kw in result["keywords"]))
        self.assertEqual(result["query"], self.query)
        self.assertEqual(result["call_count"], 1)

    @patch("hugegraph_llm.operators.llm_op.keyword_extract.LLMs")
    def test_run_with_no_llm(self, mock_llms_class):
        """Test run method with no LLM provided."""
        # Setup mock
        mock_llm = MagicMock(spec=BaseLLM)
        mock_llm.generate.return_value = (
            "KEYWORDS: artificial intelligence, machine learning, neural networks"
        )
        mock_llms_instance = MagicMock()
        mock_llms_instance.get_extract_llm.return_value = mock_llm
        mock_llms_class.return_value = mock_llms_instance

        # Create extractor with no LLM
        extractor = KeywordExtract(text=self.query)

        # Create context
        context = {}

        # Call the method
        result = extractor.run(context)

        # Verify that LLMs().get_extract_llm() was called
        mock_llms_class.assert_called_once()
        mock_llms_instance.get_extract_llm.assert_called_once()

        # Verify that llm.generate was called
        mock_llm.generate.assert_called_once()

        # Verify the result
        self.assertIn("keywords", result)
        self.assertTrue(any("artificial intelligence" in kw for kw in result["keywords"]))
        self.assertTrue(any("machine learning" in kw for kw in result["keywords"]))
        self.assertTrue(any("neural networks" in kw for kw in result["keywords"]))

    def test_run_with_no_query_in_init_but_in_context(self):
        """Test run method with no query in init but provided in context."""
        # Create extractor with no query
        extractor = KeywordExtract(llm=self.mock_llm)

        # Create context with query
        context = {"query": self.query}

        # Call the method
        result = extractor.run(context)

        # Verify the result
        self.assertIn("keywords", result)
        self.assertEqual(result["query"], self.query)

    def test_run_with_no_query_raises_assertion_error(self):
        """Test run method with no query raises assertion error."""
        # Create extractor with no query
        extractor = KeywordExtract(llm=self.mock_llm)

        # Create context with no query
        context = {}

        # Call the method and expect an assertion error
        with self.assertRaises(AssertionError) as context:
            extractor.run({})

        # Verify the assertion message
        self.assertIn("No query for keywords extraction", str(context.exception))

    @patch("hugegraph_llm.operators.llm_op.keyword_extract.LLMs")
    def test_run_with_invalid_llm_raises_assertion_error(self, mock_llms_class):
        """Test run method with invalid LLM raises assertion error."""
        # Setup mock to return an invalid LLM (not a BaseLLM instance)
        mock_llms_instance = MagicMock()
        mock_llms_instance.get_extract_llm.return_value = "not a BaseLLM instance"
        mock_llms_class.return_value = mock_llms_instance

        # Create extractor with no LLM
        extractor = KeywordExtract(text=self.query)

        # Call the method and expect an assertion error
        with self.assertRaises(AssertionError) as context:
            extractor.run({})

        # Verify the assertion message
        self.assertIn("Invalid LLM Object", str(context.exception))

    @patch("hugegraph_llm.operators.common_op.nltk_helper.NLTKHelper.stopwords")
    def test_run_with_context_parameters(self, mock_stopwords):
        """Test run method with parameters provided in context."""
        # Mock stopwords to avoid file not found error
        mock_stopwords.return_value = {"el", "la", "los", "las", "y", "en", "de"}

        # Create context with language and max_keywords
        context = {"language": "spanish", "max_keywords": 10}

        # Call the method
        result = self.extractor.run(context)

        # Verify that the parameters were updated
        self.assertEqual(self.extractor._language, "spanish")
        self.assertEqual(self.extractor._max_keywords, 10)

    def test_run_with_existing_call_count(self):
        """Test run method with existing call_count in context."""
        # Create context with existing call_count
        context = {"call_count": 5}

        # Call the method
        result = self.extractor.run(context)

        # Verify that call_count was incremented
        self.assertEqual(result["call_count"], 6)

    def test_extract_keywords_from_response_with_start_token(self):
        """Test _extract_keywords_from_response method with start token."""
        response = (
            "Some text\nKEYWORDS: artificial intelligence, machine learning, "
            "neural networks\nMore text"
        )
        keywords = self.extractor._extract_keywords_from_response(
            response, lowercase=False, start_token="KEYWORDS:"
        )

        # Check for keywords with or without leading space
        self.assertTrue(any(kw.strip() == "artificial intelligence" for kw in keywords))
        self.assertTrue(any(kw.strip() == "machine learning" for kw in keywords))
        self.assertTrue(any(kw.strip() == "neural networks" for kw in keywords))

    def test_extract_keywords_from_response_without_start_token(self):
        """Test _extract_keywords_from_response method without start token."""
        response = "artificial intelligence, machine learning, neural networks"
        keywords = self.extractor._extract_keywords_from_response(response, lowercase=False)

        # Check for keywords with or without leading space
        self.assertTrue(any(kw.strip() == "artificial intelligence" for kw in keywords))
        self.assertTrue(any(kw.strip() == "machine learning" for kw in keywords))
        self.assertTrue(any(kw.strip() == "neural networks" for kw in keywords))

    def test_extract_keywords_from_response_with_lowercase(self):
        """Test _extract_keywords_from_response method with lowercase=True."""
        response = "KEYWORDS: Artificial Intelligence, Machine Learning, Neural Networks"
        keywords = self.extractor._extract_keywords_from_response(
            response, lowercase=True, start_token="KEYWORDS:"
        )

        # Check for keywords with or without leading space
        self.assertTrue(any(kw.strip() == "artificial intelligence" for kw in keywords))
        self.assertTrue(any(kw.strip() == "machine learning" for kw in keywords))
        self.assertTrue(any(kw.strip() == "neural networks" for kw in keywords))

    def test_extract_keywords_from_response_with_multi_word_tokens(self):
        """Test _extract_keywords_from_response method with multi-word tokens."""
        # Patch NLTKHelper to return a fixed set of stopwords
        with patch(
            "hugegraph_llm.operators.llm_op.keyword_extract.NLTKHelper"
        ) as mock_nltk_helper_class:
            mock_nltk_helper = MagicMock()
            mock_nltk_helper.stopwords.return_value = {"the", "and", "of", "in"}
            mock_nltk_helper_class.return_value = mock_nltk_helper

            response = "KEYWORDS: artificial intelligence, machine learning"
            keywords = self.extractor._extract_keywords_from_response(
                response, start_token="KEYWORDS:"
            )

            # Should include both the full phrases and individual non-stopwords
            self.assertTrue(any(kw.strip() == "artificial intelligence" for kw in keywords))
            self.assertIn("artificial", keywords)
            self.assertIn("intelligence", keywords)
            self.assertTrue(any(kw.strip() == "machine learning" for kw in keywords))
            self.assertIn("machine", keywords)
            self.assertIn("learning", keywords)

    def test_extract_keywords_from_response_with_single_character_tokens(self):
        """Test _extract_keywords_from_response method with single character tokens."""
        response = "KEYWORDS: a, artificial intelligence, b, machine learning"
        keywords = self.extractor._extract_keywords_from_response(response, start_token="KEYWORDS:")

        # Single character tokens should be filtered out
        self.assertNotIn("a", keywords)
        self.assertNotIn("b", keywords)
        # Check for keywords with or without leading space
        self.assertTrue(any(kw.strip() == "artificial intelligence" for kw in keywords))
        self.assertTrue(any(kw.strip() == "machine learning" for kw in keywords))

    def test_extract_keywords_from_response_with_apostrophes(self):
        """Test _extract_keywords_from_response method with apostrophes."""
        response = "KEYWORDS: artificial intelligence, machine's learning, neural's networks"
        keywords = self.extractor._extract_keywords_from_response(response, start_token="KEYWORDS:")

        # Check for keywords with or without apostrophes and leading spaces
        self.assertTrue(any(kw.strip() == "artificial intelligence" for kw in keywords))
        self.assertTrue(any("machine" in kw and "learning" in kw for kw in keywords))
        self.assertTrue(any("neural" in kw and "networks" in kw for kw in keywords))


if __name__ == "__main__":
    unittest.main()
