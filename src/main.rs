use std::{env, error::Error, fs::read_to_string, process};

use codegen::CodeGen;
use lexer::Lexer;
use parser::Parser;

mod ast;
mod codegen;
mod lexer;
mod parser;
mod span;

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        return Err("no saya file provided".into());
    }

    let input_file = &args[1];

    let code = read_to_string(input_file)?;

    let lexer = Lexer::new(&code);
    let mut parser = Parser::new(lexer)?;
    let program = parser.parse()?;

    let mut code_gen = CodeGen::new();
    let qbe_il = code_gen.generate(&program)?;

    std::fs::write("out.ssa", qbe_il)?;

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        process::exit(1);
    }
}
