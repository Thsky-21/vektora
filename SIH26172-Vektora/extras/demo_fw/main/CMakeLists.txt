idf_component_register(
    SRCS "main.c" "ssd1306.c" "feature_extract.c"
    INCLUDE_DIRS "."
    EMBED_TXTFILES "index.html"
    PRIV_REQUIRES esp_driver_i2c esp_driver_gpio esp_wifi esp_netif esp_event nvs_flash esp_http_server lwip esp_timer)
