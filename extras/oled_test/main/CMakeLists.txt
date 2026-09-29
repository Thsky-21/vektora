idf_component_register(
    SRCS "main.cc" "model_data.cc" "ssd1306.c"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp_driver_i2c esp_driver_gpio)
