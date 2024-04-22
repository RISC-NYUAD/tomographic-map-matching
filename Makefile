TARGET = tomographic_map_matching
.PHONY: build
build:
	@docker compose build $(TARGET)

.PHONY: shell
shell:
	@docker compose run --rm $(TARGET) /bin/bash
