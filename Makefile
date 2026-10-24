.PHONY: start start-rust build-rust

# Start the audio server and open the UI in the default browser.
start:
	uv run audio-server --open

# Rust engine (same CLI, configs, UI): build the release binary, then run it.
build-rust:
	cd engine && cargo build --release -p server

start-rust: build-rust
	engine/target/release/audio-server --open
