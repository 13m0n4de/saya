use std::fs::read_to_string;

use codegen::CodeGen;
use lexer::Lexer;
use parser::Parser;

mod ast;
mod codegen;
mod lexer;
mod parser;
mod span;

fn main() {
    let code = match read_to_string("examples/hello_world.saya") {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Failed to read file: {e}");
            return;
        }
    };

    let lexer = Lexer::new(&code);
    let mut parser = match Parser::new(lexer) {
        Ok(parser) => parser,
        Err(e) => {
            eprintln!(
                "Parse error at {}:{}: {}",
                e.span.line, e.span.column, e.message
            );
            return;
        }
    };

    match parser.parse() {
        Ok(program) => {
            // println!("{program:#?}");
            let mut code_gen = CodeGen::new();
            match code_gen.generate(&program) {
                Ok(qbe_il) => {
                    println!("{qbe_il}");
                    std::fs::write("out.ssa", qbe_il).unwrap();
                }
                Err(e) => {
                    eprintln!(
                        "Code generation error at {}:{}: {}",
                        e.span.line, e.span.column, e.message
                    );
                }
            }
        }
        Err(e) => {
            eprintln!(
                "Parse error at {}:{}: {}",
                e.span.line, e.span.column, e.message
            );
        }
    }
}
