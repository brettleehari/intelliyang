"""Benchmark test queries for accuracy evaluation based on NetLLMBench."""

# Comprehensive test queries across complexity levels
TEST_QUERIES = {
    "basic_structure": [
        "What is the difference between a shelf and a slot?",
        "List the mandatory fields for device-info",
        "What are the valid card types?",
        "What is a circuit-pack in OpenROADM?",
        "Describe the port hierarchy in the device model",
    ],
    "configuration": [
        "How do I create an optical connection?",
        "What parameters are needed for amplifier configuration?",
        "Show me the steps to configure a degree",
        "How do I set up an SRG?",
        "What is required to provision a new shelf?",
    ],
    "relationships": [
        "Explain the relationship between circuit-pack and port",
        "How do SRGs relate to degrees in OpenROADM?",
        "What are the dependencies for interface configuration?",
        "How does the network model relate to the device model?",
        "What modules import org-openroadm-common-types?",
    ],
    "constraints": [
        "What are the timing constraints for OLM modules?",
        "What power limits apply to amplifier cards?",
        "Which fields are read-only vs configurable?",
        "What are the span loss limits?",
        "What validation rules apply to degree configuration?",
    ],
}

# Target accuracy thresholds
ACCURACY_TARGETS = {
    "basic_structure": 0.95,
    "configuration": 0.85,
    "relationships": 0.80,
    "constraints": 0.90,
}

# Benchmark comparison baselines
BENCHMARK_BASELINES = {
    "netconfeval_baseline": 0.58,
    "our_target": 0.85,
    "irag_benchmark": 0.975,
    "production_threshold": 0.90,
}
