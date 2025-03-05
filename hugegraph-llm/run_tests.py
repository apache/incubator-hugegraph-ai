#!/usr/bin/env python3
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

"""
Test runner script for HugeGraph-LLM.
This script sets up the environment and runs the tests.
"""

import os
import sys
import argparse
import subprocess
import nltk
from pathlib import Path


def setup_environment():
    """Set up the environment for testing."""
    # Add the project root to the Python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Download NLTK resources if needed
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    
    # Set environment variable to skip external service tests by default
    if 'HUGEGRAPH_LLM_SKIP_EXTERNAL_TESTS' not in os.environ:
        os.environ['HUGEGRAPH_LLM_SKIP_EXTERNAL_TESTS'] = 'true'
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(project_root, 'logs')
    os.makedirs(logs_dir, exist_ok=True)


def run_tests(args):
    """Run the tests with the specified arguments."""
    # Construct the pytest command
    cmd = ['pytest']
    
    # Add verbosity
    if args.verbose:
        cmd.append('-v')
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(['--cov=src/hugegraph_llm', '--cov-report=term', '--cov-report=html:coverage_html'])
    
    # Add test pattern if specified
    if args.pattern:
        cmd.append(args.pattern)
    else:
        cmd.append('src/tests')
    
    # Print the command being run
    print(f"Running: {' '.join(cmd)}")
    
    # Run the tests
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description='Run HugeGraph-LLM tests')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-c', '--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('-p', '--pattern', help='Test pattern to run (e.g., src/tests/models)')
    parser.add_argument('--external', action='store_true', help='Run tests that require external services')
    
    args = parser.parse_args()
    
    # Set up the environment
    setup_environment()
    
    # Configure external tests
    if args.external:
        os.environ['HUGEGRAPH_LLM_SKIP_EXTERNAL_TESTS'] = 'false'
        print("Running tests including those that require external services")
    else:
        print("Skipping tests that require external services (use --external to include them)")
    
    # Run the tests
    return run_tests(args)


if __name__ == '__main__':
    sys.exit(main()) 