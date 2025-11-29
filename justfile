build:
    cargo build

run file:
    cargo run {{file}}
    @cat out.ssa

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
    -rm -f out.ssa out.s a.out

compile file: (run file)
    qbe out.ssa -o out.s
    cc out.s -o a.out

exec file: (compile file)
    -./a.out; echo "Exit code: $?"
