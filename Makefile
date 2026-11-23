.PHONY: start

# Start the audio server and open the UI in the default browser.
#   make start                  configs/main.yaml (OSC to localhost)
#   make start CONFIG=pi        use configs/pi.yaml instead (e.g. + the Pi as OSC destination)
CONFIG ?= main

# main is the default config: pass no --config, so a missing main.yaml is
# seeded from main.example.yaml.
CONFIG_ARG := $(if $(filter main,$(CONFIG)),,--config configs/$(CONFIG).yaml)

start:
	uv run audio-server --open $(CONFIG_ARG)
