.PHONY: start

# Start the audio server and open the UI in the default browser.
#   make start                  Rust engine on configs/main.yaml (OSC to localhost)
#   make start CONFIG=pi        use configs/pi.yaml instead (e.g. + the Pi as OSC destination)
#   make start ENGINE=python    the Python reference engine instead (same CLI/configs/UI)
CONFIG ?= main
ENGINE ?= rust

# main is the default config: pass no --config, so a missing main.yaml is
# seeded from main.example.yaml.
CONFIG_ARG := $(if $(filter main,$(CONFIG)),,--config configs/$(CONFIG).yaml)

start:
ifeq ($(ENGINE),python)
	uv run audio-server --open $(CONFIG_ARG)
else
	cd engine && cargo build --release -p server
	engine/target/release/audio-server --open $(CONFIG_ARG)
endif
