from weather_collector.config import Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.app_name == "Weather Collector"
