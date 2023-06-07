def pytest_configure(config):
    config.addinivalue_line(
        "markers", "skip_plotting: skip the test if it system does not support plotting"
    )
