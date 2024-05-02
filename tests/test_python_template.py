#!/usr/bin/env python3

# Script Description ##########################################################
"""
Test functions for fema-open-source-temp.

To run the tests, try the following commands:

    $ pytest 
    $ pytest tests/test_python_template.py -vv
"""

import os
import pytest

from src.example_function import (
    example_function,
)

@pytest.fixture(scope="module")
def low_and_high():
    """Create reusable parameters for testing. These parameters
    are available in the namespace of the test functions.
    """

    # create low and high parameters (integers)
    low = 0
    high = 10
    
    yield (low, high)


@pytest.fixture(scope="module")
def low_high_increment():
    """Create reusable parameters for testing. These parameters
    are available in the namespace of the test functions.
    """

    # create low and high parameters (integers)
    low = 0
    high = 20
    increment = 10
    
    yield (low, high, increment)


def test_example_function(low_and_high: tuple):
    """Test the example function"""

    # call the example function
    new_list = example_function(low_and_high[0], low_and_high[1])

    if new_list !=  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        raise RuntimeError("example_function did not return the expected list")
