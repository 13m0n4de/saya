build:
    cargo build

run:
    cargo run

check:
    cargo check

test:
    cargo test

fmt:
    cargo fmt

lint:
    cargo clippy

clean:
    cargo clean
    rm out.ssa out.s a.out

compile: run
    qbe out.ssa -o out.s
    cc out.s -o a.out

exec: compile
    -./a.out; echo "Exit code: $?"
