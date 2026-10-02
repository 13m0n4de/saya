set positional-arguments

# just exec foo.saya
# echo '...' | just exec
exec *args:
    #!/usr/bin/env bash
    set -uo pipefail

    exe=$(mktemp)
    trap 'rm -f "$exe"' EXIT

    ir=$(cargo run -q -- "$@") || exit 1
    asm=$(qbe <<< "$ir") || exit 1
    cc -x assembler - -o "$exe" <<< "$asm" || exit 1

    "$exe"; echo "exit code: $?"
