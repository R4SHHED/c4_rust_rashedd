use std::collections::HashMap;
use std::env;
use std::fs;
use std::iter::Peekable;
use std::slice::Iter;

// Define the types of tokens our compiler can recognize
#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Int,                  // 'int' keyword
    Return,               // 'return' keyword
    Identifier(String),   // Variable or function names
    Number(i64),          // Integer literals
    StringLiteral(String), // String literals like "hello"
    LParen,               // '('
    RParen,               // ')'
    LBrace,               // '{'
    RBrace,               // '}'
    Semicolon,            // ';'
    Plus,                 // '+'
    Star,                 // '*'
    Minus,                // '-'
    Divide,               // '/'
    Mod,                  // '%'
    Equal,                // '=='
    Less,                 // '<'
    Greater,              // '>'
    If,                   // 'if' keyword
    Else,                 // 'else' keyword
    While,                // 'while' keyword
    Assign,               // '='
    Comma,                // ','
    Unknown(char),        // Any unrecognized character
}

// Breaks source code into tokens (like words in a sentence)
pub fn lexify_source(source: &str) -> Vec<Token> {
    let mut tokens = Vec::new(); // Store our tokens here
    let mut chars = source.chars().peekable(); // Peekable iterator to look ahead

    // Loop through each character in the source code
    while let Some(&ch) = chars.peek() {
        match ch {
            // Skip whitespace characters
            ' ' | '\n' | '\r' | '\t' => {
                chars.next();
            }
            // Handle string literals (e.g., "hello\n")
            '"' => {
                chars.next(); // Skip opening quote
                let mut s = String::new(); // Build the string
                while let Some(c) = chars.peek() {
                    if *c == '"' {
                        chars.next(); // Skip closing quote
                        break;
                    }
                    if *c == '\\' {
                        chars.next(); // Skip backslash
                        if let Some(next) = chars.peek() {
                            match next {
                                'n' => { s.push('\n'); chars.next(); } // Newline
                                't' => { s.push('\t'); chars.next(); } // Tab
                                '\\' => { s.push('\\'); chars.next(); } // Literal backslash
                                '"' => { s.push('"'); chars.next(); } // Literal quote
                                _ => { s.push('\\'); s.push(*next); chars.next(); } // Other escapes
                            }
                        }
                    } else {
                        s.push(*c); // Add character to string
                        chars.next();
                    }
                }
                tokens.push(Token::StringLiteral(s)); // Store string token
            }
            // Handle single-character tokens
            '(' => { chars.next(); tokens.push(Token::LParen); }
            ')' => { chars.next(); tokens.push(Token::RParen); }
            '{' => { chars.next(); tokens.push(Token::LBrace); }
            '}' => { chars.next(); tokens.push(Token::RBrace); }
            ';' => { chars.next(); tokens.push(Token::Semicolon); }
            '+' => { chars.next(); tokens.push(Token::Plus); }
            '*' => { chars.next(); tokens.push(Token::Star); }
            '-' => { chars.next(); tokens.push(Token::Minus); }
            '/' => { chars.next(); tokens.push(Token::Divide); }
            '%' => { chars.next(); tokens.push(Token::Mod); }
            // Handle '=' (assignment) or '==' (equality)
            '=' => {
                chars.next();
                if let Some('=') = chars.peek() {
                    chars.next();
                    tokens.push(Token::Equal);
                } else {
                    tokens.push(Token::Assign);
                }
            }
            '<' => { chars.next(); tokens.push(Token::Less); }
            '>' => { chars.next(); tokens.push(Token::Greater); }
            ',' => { chars.next(); tokens.push(Token::Comma); }
            // Handle identifiers and keywords
            'a'..='z' | 'A'..='Z' | '_' => {
                let mut ident = String::new();
                while let Some(c) = chars.peek() {
                    if c.is_alphanumeric() || *c == '_' {
                        ident.push(*c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                // Map keywords to specific tokens
                match ident.as_str() {
                    "int" => tokens.push(Token::Int),
                    "return" => tokens.push(Token::Return),
                    "if" => tokens.push(Token::If),
                    "else" => tokens.push(Token::Else),
                    "while" => tokens.push(Token::While),
                    "printf" => tokens.push(Token::Identifier(ident)), // Treat printf as identifier
                    _ => tokens.push(Token::Identifier(ident)),
                }
            }
            // Handle numbers
            '0'..='9' => {
                let mut num = 0;
                while let Some(c) = chars.peek() {
                    if c.is_digit(10) {
                        num = num * 10 + c.to_digit(10).unwrap() as i64;
                        chars.next();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Number(num));
            }
            // Store unrecognized characters
            _ => {
                tokens.push(Token::Unknown(ch));
                chars.next();
            }
        }
    }

    tokens // Return the list of tokens
}

// Represents nodes in the Abstract Syntax Tree (AST)
#[derive(Debug, PartialEq)]
pub enum ASTNode {
    Return(Box<Expr>), // Return statement
    If {
        condition: Box<Expr>, // Condition to check
        then_branch: Box<ASTNode>, // Code if true
        else_branch: Option<Box<ASTNode>>, // Code if false (optional)
    },
    While {
        condition: Box<Expr>, // Loop condition
        body: Box<ASTNode>, // Loop body
    },
    Sequence(Vec<ASTNode>), // List of statements
    Declaration(String, Box<Expr>), // Variable declaration
    Assignment(String, Box<Expr>), // Variable assignment
    FunctionDef {
        name: String, // Function name
        params: Vec<String>, // Parameter names
        body: Box<ASTNode>, // Function body
    },
}

// Represents expressions in the AST
#[derive(Debug, PartialEq)]
pub enum Expr {
    Number(i64), // Integer literal
    StringLiteral(String), // String literal
    Variable(String), // Variable reference
    Add(Box<Expr>, Box<Expr>), // Addition
    Sub(Box<Expr>, Box<Expr>), // Subtraction
    Mul(Box<Expr>, Box<Expr>), // Multiplication
    Div(Box<Expr>, Box<Expr>), // Division
    Mod(Box<Expr>, Box<Expr>), // Modulus
    Equal(Box<Expr>, Box<Expr>), // Equality check
    Less(Box<Expr>, Box<Expr>), // Less than
    Greater(Box<Expr>, Box<Expr>), // Greater than
    Call(String, Vec<Expr>), // Function call
    Var(String), // Variable reference
}

// Converts tokens into an AST
pub fn build_ast(tokens: &[Token]) -> ASTNode {
    let mut iter = tokens.iter().peekable();
    
    // Expect a function like 'int main() {'
    match (iter.next(), iter.next(), iter.next()) {
        (Some(Token::Int), Some(Token::Identifier(_)), Some(Token::LParen)) => {
            while let Some(token) = iter.next() {
                if *token == Token::LBrace {
                    break; // Found the opening brace
                }
            }
        }
        _ => panic!("Invalid function declaration"),
    }

    let mut statements = Vec::new(); // Collect statements

    // Parse statements until we hit a closing brace
    while let Some(token) = iter.peek() {
        match token {
            Token::Return | Token::If | Token::While | Token::LBrace | Token::Int | Token::Identifier(_) => {
                statements.push(construct_statement(&mut iter));
            }
            Token::RBrace => {
                iter.next();
                break;
            }
            _ => panic!("Unexpected token in block: {:?}", token),
        }
    }

    ASTNode::Sequence(statements) // Return sequence of statements
}

// Parses a variable declaration (e.g., int x = 5;)
fn create_var_def(iter: &mut Peekable<Iter<Token>>) -> ASTNode {
    // Expect an identifier for the variable name
    let name = match iter.next() {
        Some(Token::Identifier(name)) => name.clone(),
        _ => panic!("Expected variable name"),
    };

    check_token(iter, Token::Assign); // Expect '='
    let expr = parse_expr(iter); // Parse the expression
    check_token(iter, Token::Semicolon); // Expect ';'

    ASTNode::Declaration(name, expr) // Return declaration node
}

// Parses an assignment statement (e.g., x = 5;)
fn assign_variable(iter: &mut Peekable<Iter<Token>>) -> ASTNode {
    // Expect an identifier
    let name = match iter.next() {
        Some(Token::Identifier(name)) => name.clone(),
        _ => panic!("Expected variable"),
    };

    check_token(iter, Token::Assign); // Expect '='
    let expr = parse_expr(iter); // Parse the expression
    check_token(iter, Token::Semicolon); // Expect ';'

    ASTNode::Assignment(name, expr) // Return assignment node
}

// Parses a single statement
fn construct_statement(iter: &mut Peekable<Iter<Token>>) -> ASTNode {
    match iter.peek() {
        Some(Token::Return) => {
            iter.next(); // Skip 'return'
            let expr = parse_expr(iter);
            check_token(iter, Token::Semicolon);
            ASTNode::Return(expr)
        }
        Some(Token::If) => {
            iter.next(); // Skip 'if'
            process_if_block(iter)
        }
        Some(Token::LBrace) => {
            process_code_block(iter)
        }
        Some(Token::While) => {
            iter.next(); // Skip 'while'
            create_while_loop(iter)
        }
        Some(Token::Int) => {
            iter.next(); // Skip 'int'
            create_var_def(iter)
        }
        Some(Token::Identifier(_)) => assign_variable(iter),
        _ => panic!("Expected a valid statement"),
    }
}

// Parses a while loop
fn create_while_loop(iter: &mut Peekable<Iter<Token>>) -> ASTNode {
    check_token(iter, Token::LParen); // Expect '('
    let condition = parse_expr(iter); // Parse condition
    check_token(iter, Token::RParen); // Expect ')'

    let body = construct_statement(iter); // Parse loop body

    ASTNode::While {
        condition,
        body: Box::new(body),
    }
}

// Parses a block of statements (e.g., { ... })
fn process_code_block(iter: &mut Peekable<Iter<Token>>) -> ASTNode {
    check_token(iter, Token::LBrace); // Expect '{'
    let mut stmts = Vec::new();

    // Parse statements until '}'
    while let Some(token) = iter.peek() {
        match token {
            Token::RBrace => {
                iter.next();
                break;
            }
            Token::Return | Token::If | Token::While | Token::LBrace | Token::Int | Token::Identifier(_) => {
                stmts.push(construct_statement(iter));
            }
            t => panic!("Unexpected token in block: {:?}", t),
        }
    }

    ASTNode::Sequence(stmts)
}

// Parses an if statement
fn process_if_block(iter: &mut Peekable<Iter<Token>>) -> ASTNode {
    check_token(iter, Token::LParen); // Expect '('
    let condition = parse_expr(iter); // Parse condition
    check_token(iter, Token::RParen); // Expect ')'

    let then_branch = construct_statement(iter); // Parse then branch

    // Check for optional else branch
    let else_branch = if let Some(Token::Else) = iter.peek() {
        iter.next();
        Some(Box::new(construct_statement(iter)))
    } else {
        None
    };

    ASTNode::If {
        condition,
        then_branch: Box::new(then_branch),
        else_branch,
    }
}

// Checks if the next token matches the expected token
fn check_token(iter: &mut Peekable<Iter<Token>>, expected: Token) {
    match iter.next() {
        Some(t) if *t == expected => {}, // Token matches, proceed
        other => panic!("Expected {:?}, got {:?}", expected, other),
    }
}

// Parses an expression (top-level)
fn parse_expr(iter: &mut Peekable<Iter<Token>>) -> Box<Expr> {
    evaluate_comparison(iter)
}

// Parses comparison expressions (e.g., a == b, a < b)
fn evaluate_comparison(iter: &mut Peekable<Iter<Token>>) -> Box<Expr> {
    let mut left = compute_arithmetic(iter); // Parse lower-precedence expression

    // Handle comparison operators
    while let Some(token) = iter.peek() {
        match token {
            Token::Equal => {
                iter.next();
                let right = compute_arithmetic(iter);
                left = Box::new(Expr::Equal(left, right));
            }
            Token::Less => {
                iter.next();
                let right = compute_arithmetic(iter);
                left = Box::new(Expr::Less(left, right));
            }
            Token::Greater => {
                iter.next();
                let right = compute_arithmetic(iter);
                left = Box::new(Expr::Greater(left, right));
            }
            _ => break,
        }
    }

    left
}

// Parses addition and subtraction
fn compute_arithmetic(iter: &mut Peekable<Iter<Token>>) -> Box<Expr> {
    let mut left = perform_multiplication(iter); // Parse higher-precedence expression

    // Handle + and -
    while let Some(token) = iter.peek() {
        match token {
            Token::Plus => {
                iter.next();
                let right = perform_multiplication(iter);
                left = Box::new(Expr::Add(left, right));
            }
            Token::Minus => {
                iter.next();
                let right = perform_multiplication(iter);
                left = Box::new(Expr::Sub(left, right));
            }
            _ => break,
        }
    }

    left
}

// Parses multiplication, division, and modulus
fn perform_multiplication(iter: &mut Peekable<Iter<Token>>) -> Box<Expr> {
    let mut left = extract_primary(iter); // Parse primary expression

    // Handle *, /, and %
    while let Some(token) = iter.peek() {
        match token {
            Token::Star => {
                iter.next();
                let right = extract_primary(iter);
                left = Box::new(Expr::Mul(left, right));
            }
            Token::Divide => {
                iter.next();
                let right = extract_primary(iter);
                left = Box::new(Expr::Div(left, right));
            }
            Token::Mod => {
                iter.next();
                let right = extract_primary(iter);
                left = Box::new(Expr::Mod(left, right));
            }
            _ => break,
        }
    }

    left
}

// Parses primary expressions (numbers, strings, variables, calls)
fn extract_primary(iter: &mut Peekable<Iter<Token>>) -> Box<Expr> {
    match iter.next() {
        Some(Token::Number(n)) => Box::new(Expr::Number(*n)),
        Some(Token::StringLiteral(s)) => Box::new(Expr::StringLiteral(s.clone())),
        Some(Token::Identifier(name)) => {
            let name = name.clone();
            // Check if this is a function call
            if let Some(Token::LParen) = iter.peek() {
                iter.next(); // Skip '('
                let mut args = Vec::new();
                while let Some(token) = iter.peek() {
                    if let Token::RParen = token {
                        break;
                    }
                    let arg = parse_expr(iter);
                    args.push(*arg);
                    if let Some(Token::Comma) = iter.peek() {
                        iter.next();
                    } else {
                        break;
                    }
                }
                check_token(iter, Token::RParen);
                Box::new(Expr::Call(name, args))
            } else {
                Box::new(Expr::Var(name))
            }
        }
        Some(Token::LParen) => {
            let expr = parse_expr(iter); // Parse expression inside parentheses
            match iter.next() {
                Some(Token::RParen) => expr,
                _ => panic!("Expected closing parenthesis"),
            }
        }
        other => panic!("Expected number, string, variable, or '(', got {:?}", other),
    }
}

// Instruction set for the virtual machine
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Instruction {
    IMM(i64),    // Push immediate value
    PSH,         // Push top of stack
    ADD,         // Add top two values
    SUB,         // Subtract top two values
    MUL,         // Multiply top two values
    DIV,         // Divide top two values
    MOD,         // Modulus of top two values
    JMP(usize),  // Jump to address
    BZ(usize),   // Branch if zero
    BNZ(usize),  // Branch if non-zero
    JSR(usize),  // Jump to subroutine
    ENT(usize),  // Enter function
    ADJ(usize),  // Adjust stack
    LEV,         // Leave function
    LEA(usize),  // Load effective address
    LI,          // Load integer
    LC,          // Load character
    SI,          // Store integer
    SC,          // Store character
    EXIT,        // Exit program
    PRTF,        // Print string
    MALC,        // Allocate memory
    FREE,        // Free memory
    MSET,        // Set memory
    MCMP,        // Compare memory
    OPEN,        // Open file
    READ,        // Read file
    CLOS,        // Close file
    EQ,          // Equality check
    LT,          // Less than
    GT,          // Greater than
}

// Stack-based virtual machine
pub struct VM {
    pub stack: Vec<i64>,         // Stack for computations
    pub pc: usize,               // Program counter
    pub bp: usize,               // Base pointer
    pub program: Vec<Instruction>, // Program instructions
    pub running: bool,           // VM running state
    pub data_segment: Vec<String>, // String literals
}

impl VM {
    // Create a new VM instance
    pub fn new(program: Vec<Instruction>, data_segment: Vec<String>) -> Self {
        VM {
            stack: Vec::new(),
            pc: 0,
            bp: 0,
            program,
            running: true,
            data_segment,
        }
    }

    // Execute the VM program
    pub fn execute_bytecode(&mut self) {
        while self.running {
            if self.pc >= self.program.len() {
                panic!("Program counter out of bounds");
            }

            // Process each instruction
            match self.program[self.pc] {
                Instruction::IMM(val) => {
                    self.stack.push(val); // Push value onto stack
                }
                Instruction::PSH => {
                    if let Some(&top) = self.stack.last() {
                        self.stack.push(top); // Duplicate top of stack
                    } else {
                        panic!("PSH failed: stack is empty");
                    }
                }
                Instruction::ADD => {
                    let b = self.stack.pop().expect("ADD: missing operand B");
                    let a = self.stack.pop().expect("ADD: missing operand A");
                    self.stack.push(a + b); // Add and push result
                }
                Instruction::SUB => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    self.stack.push(a - b); // Subtract and push result
                }
                Instruction::MUL => {
                    let b = self.stack.pop().expect("MUL: missing operand B");
                    let a = self.stack.pop().expect("MUL: missing operand A");
                    self.stack.push(a * b); // Multiply and push result
                }
                Instruction::DIV => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    self.stack.push(a / b); // Divide and push result
                }
                Instruction::MOD => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    self.stack.push(a % b); // Modulus and push result
                }
                Instruction::JMP(target) => {
                    self.pc = target; // Jump to target address
                    continue;
                }
                Instruction::BZ(target) => {
                    let cond = self.stack.pop().unwrap();
                    if cond == 0 {
                        self.pc = target; // Jump if condition is zero
                        continue;
                    }
                }
                Instruction::BNZ(target) => {
                    let cond = self.stack.pop().unwrap();
                    if cond != 0 {
                        self.pc = target; // Jump if condition is non-zero
                        continue;
                    }
                }
                Instruction::JSR(target) => {
                    self.stack.push((self.pc + 1) as i64); // Save return address
                    self.pc = target; // Jump to subroutine
                    continue;
                }
                Instruction::ENT(size) => {
                    self.stack.push(self.bp as i64); // Save base pointer
                    self.bp = self.stack.len(); // Set new base pointer
                    self.stack.resize(self.stack.len() + size, 0); // Allocate locals
                }
                Instruction::ADJ(n) => {
                    for _ in 0..n {
                        self.stack.pop(); // Remove n items from stack
                    }
                }
                Instruction::LEV => {
                    let old_bp = self.stack[self.bp - 1];
                    self.stack.truncate(self.bp - 1); // Restore stack
                    self.bp = old_bp as usize; // Restore base pointer
                    self.pc = self.stack.pop().unwrap() as usize; // Return
                    continue;
                }
                Instruction::LEA(offset) => {
                    let addr = self.bp + offset;
                    self.stack.push(addr as i64); // Push address
                }
                Instruction::LI => {
                    let addr = self.stack.pop().unwrap() as usize;
                    let val = self.stack[addr];
                    self.stack.push(val); // Load value from address
                }
                Instruction::LC => {
                    let addr = self.stack.pop().unwrap() as usize;
                    let val = self.stack[addr] & 0xFF;
                    self.stack.push(val); // Load byte from address
                }
                Instruction::SI => {
                    let val = self.stack.pop().unwrap();
                    let addr = self.stack.pop().unwrap() as usize;
                    self.stack[addr] = val; // Store value at address
                }
                Instruction::SC => {
                    let val = self.stack.pop().unwrap() & 0xFF;
                    let addr = self.stack.pop().unwrap() as usize;
                    self.stack[addr] = val; // Store byte at address
                }
                Instruction::EXIT => {
                    if let Some(&result) = self.stack.last() {
                        println!("exit({})", result); // Print exit code
                    } else {
                        println!("Program exited without a return value");
                    }
                    self.running = false; // Stop VM
                }
                Instruction::PRTF => {
                    let arg_count = self.stack.pop().unwrap() as usize;
                    let fmt_addr = self.stack.pop().unwrap() as usize;
                    if fmt_addr < self.data_segment.len() {
                        let fmt_str = &self.data_segment[fmt_addr];
                        print!("{}", fmt_str); // Print string
                        self.stack.push(fmt_str.len() as i64); // Push string length
                    } else {
                        panic!("Invalid format string address: {}", fmt_addr);
                    }
                }
                Instruction::MALC => {
                    self.stack.push(0x1000); // Simulate memory allocation
                }
                Instruction::FREE => {
                    let _ = self.stack.pop(); // Simulate freeing memory
                }
                Instruction::MSET => {
                    let _ = self.stack.pop();
                    let _ = self.stack.pop();
                    let _ = self.stack.pop(); // Simulate memory set
                }
                Instruction::MCMP => {
                    let _ = self.stack.pop();
                    let _ = self.stack.pop();
                    let _ = self.stack.pop();
                    self.stack.push(0); // Simulate memory compare
                }
                Instruction::OPEN => {
                    let _ = self.stack.pop();
                    let _ = self.stack.pop();
                    self.stack.push(3); // Simulate file open
                }
                Instruction::READ => {
                    let _ = self.stack.pop();
                    let _ = self.stack.pop();
                    let _ = self.stack.pop();
                    self.stack.push(10); // Simulate file read
                }
                Instruction::CLOS => {
                    let _ = self.stack.pop();
                    self.stack.push(0); // Simulate file close
                }
                Instruction::EQ => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    self.stack.push((a == b) as i64); // Push equality result
                }
                Instruction::LT => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    self.stack.push((a < b) as i64); // Push less-than result
                }
                Instruction::GT => {
                    let b = self.stack.pop().unwrap();
                    let a = self.stack.pop().unwrap();
                    self.stack.push((a > b) as i64); // Push greater-than result
                }
            }
            self.pc += 1; // Move to next instruction
        }
    }
}

// Generate VM instructions from AST
pub fn generate_bytecode(ast: &ASTNode) -> (Vec<Instruction>, Vec<String>) {
    let mut instructions = Vec::new(); // Store generated instructions
    let mut symbol_table = HashMap::new(); // Map variables to stack offsets
    let mut next_offset = 0; // Track stack offsets
    let mut patches: Vec<(usize, String)> = Vec::new(); // Track function calls
    let mut data_segment: Vec<String> = Vec::new(); // Store string literals
    let mut string_to_addr: HashMap<String, usize> = HashMap::new(); // Map strings to addresses

    instructions.push(Instruction::ENT(0)); // Initialize stack frame

    // Generate instructions recursively
    assemble_instructions(
        ast,
        &mut instructions,
        &mut symbol_table,
        &mut next_offset,
        &mut patches,
        &mut data_segment,
        &mut string_to_addr,
    );

    // Map function names to their instruction indices
    let mut function_addresses = HashMap::new();
    if let ASTNode::Sequence(stmts) = ast {
        for (i, stmt) in stmts.iter().enumerate() {
            if let ASTNode::FunctionDef { name, .. } = stmt {
                function_addresses.insert(name.clone(), i);
            }
        }
    }

    // Resolve function calls
    for (index, func_name) in patches {
        if let Some(&addr) = function_addresses.get(&func_name) {
            instructions[index] = Instruction::JSR(addr);
        } else if func_name == "printf" {
            instructions[index] = Instruction::PRTF; // Map printf to PRTF
        } else {
            panic!("Unresolved function call: {}", func_name);
        }
    }

    instructions[0] = Instruction::ENT(next_offset); // Set correct frame size

    (instructions, data_segment) // Return instructions and data segment
}

// Recursively generate instructions from AST
fn assemble_instructions(
    ast: &ASTNode,
    instructions: &mut Vec<Instruction>,
    symbol_table: &mut HashMap<String, usize>,
    next_offset: &mut usize,
    patches: &mut Vec<(usize, String)>,
    data_segment: &mut Vec<String>,
    string_to_addr: &mut HashMap<String, usize>,
) {
    match ast {
        ASTNode::Return(expr) => {
            emit_expression(expr, instructions, symbol_table, patches, data_segment, string_to_addr);
            instructions.push(Instruction::PSH); // Push return value
            instructions.push(Instruction::EXIT); // Exit program
        }
        ASTNode::If {
            condition,
            then_branch,
            else_branch,
        } => {
            // Generate condition code
            emit_expression(
                condition,
                instructions,
                symbol_table,
                patches,
                data_segment,
                string_to_addr,
            );
            let jump_false_index = instructions.len();
            instructions.push(Instruction::BZ(9999)); // Placeholder jump
            // Generate then branch
            assemble_instructions(
                then_branch,
                instructions,
                symbol_table,
                next_offset,
                patches,
                data_segment,
                string_to_addr,
            );
            if let Some(else_branch) = else_branch {
                let jump_over_else_index = instructions.len();
                instructions.push(Instruction::JMP(9999)); // Placeholder jump
                let else_start = instructions.len();
                // Generate else branch
                assemble_instructions(
                    else_branch,
                    instructions,
                    symbol_table,
                    next_offset,
                    patches,
                    data_segment,
                    string_to_addr,
                );
                let after_else = instructions.len();
                instructions[jump_false_index] = Instruction::BZ(else_start);
                instructions[jump_over_else_index] = Instruction::JMP(after_else);
            } else {
                let after_then = instructions.len();
                instructions[jump_false_index] = Instruction::BZ(after_then);
            }
        }
        ASTNode::While { condition, body } => {
            let loop_start = instructions.len();
            // Generate condition code
            emit_expression(
                condition,
                instructions,
                symbol_table,
                patches,
                data_segment,
                string_to_addr,
            );
            let jump_if_false_index = instructions.len();
            instructions.push(Instruction::BZ(9999)); // Placeholder jump
            // Generate loop body
            assemble_instructions(
                body,
                instructions,
                symbol_table,
                next_offset,
                patches,
                data_segment,
                string_to_addr,
            );
            instructions.push(Instruction::JMP(loop_start)); // Loop back
            let loop_end = instructions.len();
            instructions[jump_if_false_index] = Instruction::BZ(loop_end);
        }
        ASTNode::Sequence(statements) => {
            // Process each statement
            for stmt in statements {
                assemble_instructions(
                    stmt,
                    instructions,
                    symbol_table,
                    next_offset,
                    patches,
                    data_segment,
                    string_to_addr,
                );
            }
        }
        ASTNode::Declaration(name, expr) => {
            let offset = *next_offset;
            *next_offset += 1; // Allocate stack slot
            symbol_table.insert(name.clone(), offset);
            instructions.push(Instruction::LEA(offset)); // Load address
            emit_expression(expr, instructions, symbol_table, patches, data_segment, string_to_addr);
            instructions.push(Instruction::SI); // Store value
        }
        ASTNode::Assignment(name, expr) => {
            if let Some(&offset) = symbol_table.get(name) {
                instructions.push(Instruction::LEA(offset)); // Load address
                emit_expression(expr, instructions, symbol_table, patches, data_segment, string_to_addr);
                instructions.push(Instruction::SI); // Store value
            } else {
                panic!("Assignment to undeclared variable: {}", name);
            }
        }
        ASTNode::FunctionDef { name: _, params, body } => {
            symbol_table.clear(); // Clear symbols for new scope
            *next_offset = params.len(); // Allocate slots for params
            for (i, param) in params.iter().enumerate() {
                symbol_table.insert(param.clone(), i); // Map params to offsets
            }
            assemble_instructions(
                body,
                instructions,
                symbol_table,
                next_offset,
                patches,
                data_segment,
                string_to_addr,
            );
        }
    }
}

// Generate instructions for an expression
fn emit_expression(
    expr: &Expr,
    instructions: &mut Vec<Instruction>,
    symbol_table: &HashMap<String, usize>,
    patches: &mut Vec<(usize, String)>,
    data_segment: &mut Vec<String>,
    string_to_addr: &mut HashMap<String, usize>,
) {
    match expr {
        Expr::Number(n) => {
            instructions.push(Instruction::IMM(*n)); // Push number
        }
        Expr::StringLiteral(s) => {
            // Store string in data segment
            let addr = if let Some(&addr) = string_to_addr.get(s) {
                addr
            } else {
                let addr = data_segment.len();
                data_segment.push(s.clone());
                string_to_addr.insert(s.clone(), addr);
                addr
            };
            instructions.push(Instruction::IMM(addr as i64)); // Push string address
        }
        Expr::Add(lhs, rhs) => {
            emit_expression(lhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            emit_expression(rhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            instructions.push(Instruction::ADD);
        }
        Expr::Sub(lhs, rhs) => {
            emit_expression(lhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            emit_expression(rhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            instructions.push(Instruction::SUB);
        }
        Expr::Mul(lhs, rhs) => {
            emit_expression(lhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            emit_expression(rhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            instructions.push(Instruction::MUL);
        }
        Expr::Div(lhs, rhs) => {
            emit_expression(lhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            emit_expression(rhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            instructions.push(Instruction::DIV);
        }
        Expr::Mod(lhs, rhs) => {
            emit_expression(lhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            emit_expression(rhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            instructions.push(Instruction::MOD);
        }
        Expr::Equal(lhs, rhs) => {
            emit_expression(lhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            emit_expression(rhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            instructions.push(Instruction::EQ);
        }
        Expr::Less(lhs, rhs) => {
            emit_expression(lhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            emit_expression(rhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            instructions.push(Instruction::LT);
        }
        Expr::Greater(lhs, rhs) => {
            emit_expression(lhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            emit_expression(rhs, instructions, symbol_table, patches, data_segment, string_to_addr);
            instructions.push(Instruction::GT);
        }
        Expr::Variable(name) => {
            if let Some(&offset) = symbol_table.get(name) {
                instructions.push(Instruction::LEA(offset));
                instructions.push(Instruction::LI); // Load variable value
            } else {
                panic!("Use of undeclared variable: {}", name);
            }
        }
        Expr::Call(func_name, args) => {
            // Push arguments in order
            for arg in args {
                emit_expression(arg, instructions, symbol_table, patches, data_segment, string_to_addr);
            }
            let placeholder_index = instructions.len();
            instructions.push(Instruction::JSR(9999)); // Placeholder for function call
            patches.push((placeholder_index, func_name.clone()));
        }
        Expr::Var(name) => {
            if let Some(&offset) = symbol_table.get(name) {
                instructions.push(Instruction::LEA(offset));
                instructions.push(Instruction::LI); // Load variable value
            } else {
                panic!("Use of undeclared variable: {}", name);
            }
        }
    }
}

// Main function to run the compiler
fn main() {
    let args: Vec<String> = env::args().collect(); // Get command-line args

    // Check if input file is provided
    if args.len() != 2 {
        eprintln!("Usage: {} <input.c>", args[0]);
        std::process::exit(1);
    }

    let filename = &args[1];
    let source = fs::read_to_string(filename).expect("Failed to read source file");

    // Compile and run the program
    let tokens = lexify_source(&source);
    let ast = build_ast(&tokens);
    let (program, data_segment) = generate_bytecode(&ast);
    let mut vm = VM::new(program, data_segment);
    vm.execute_bytecode();
}

// Tests for the compiler
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_function_tokenization() {
        let src = "int main() { return 42; }";
        let tokens = lexify_source(src);
        assert_eq!(
            tokens,
            vec![
                Token::Int,
                Token::Identifier("main".into()),
                Token::LParen,
                Token::RParen,
                Token::LBrace,
                Token::Return,
                Token::Number(42),
                Token::Semicolon,
                Token::RBrace
            ]
        );
    }

    #[test]
    fn test_vm_addition() {
        let prog = vec![
            Instruction::IMM(4),
            Instruction::IMM(6),
            Instruction::ADD,
            Instruction::EXIT,
        ];
        let mut vm = VM::new(prog, vec![]);
        vm.execute_bytecode();
        assert_eq!(vm.stack, vec![10]);
    }

    #[test]
    fn test_vm_branch_zero() {
        let prog = vec![
            Instruction::IMM(0),
            Instruction::BZ(5),
            Instruction::IMM(100),
            Instruction::IMM(200),
            Instruction::ADD,
            Instruction::IMM(9),
            Instruction::EXIT,
        ];
        let mut vm = VM::new(prog, vec![]);
        vm.execute_bytecode();
        assert_eq!(vm.stack, vec![9]);
    }

    #[test]
    fn test_vm_branch_nonzero() {
        let prog = vec![
            Instruction::IMM(1),
            Instruction::BNZ(5),
            Instruction::IMM(10),
            Instruction::IMM(20),
            Instruction::ADD,
            Instruction::IMM(7),
            Instruction::EXIT,
        ];
        let mut vm = VM::new(prog, vec![]);
        vm.execute_bytecode();
        assert_eq!(vm.stack, vec![7]);
    }

    #[test]
    fn test_vm_function_call() {
        let prog = vec![
            Instruction::JSR(4),
            Instruction::IMM(123),
            Instruction::PSH,
            Instruction::EXIT,
            Instruction::ENT(0),
            Instruction::LEV,
        ];
        let mut vm = VM::new(prog, vec![]);
        vm.execute_bytecode();
        assert_eq!(vm.stack.last(), Some(&123));
    }

    #[test]
    fn test_vm_memory_access() {
        let prog = vec![
            Instruction::ENT(1),
            Instruction::LEA(0),
            Instruction::IMM(888),
            Instruction::SI,
            Instruction::LEA(0),
            Instruction::LI,
            Instruction::EXIT,
        ];
        let mut vm = VM::new(prog, vec![]);
        vm.execute_bytecode();
        assert_eq!(vm.stack.last(), Some(&888));
    }

    #[test]
    fn test_sum_expression_parsing() {
        let tokens = lexify_source("int main() { return 8 + 9; }");
        let ast = build_ast(&tokens);
        assert_eq!(
            ast,
            ASTNode::Sequence(vec![ASTNode::Return(Box::new(Expr::Add(
                Box::new(Expr::Number(8)),
                Box::new(Expr::Number(9)),
            )))]),
        );
    }

    #[test]
    fn test_nested_multiplication_parsing() {
        let tokens = lexify_source("int main() { return (2 + 3) * 4; }");
        let ast = build_ast(&tokens);
        assert_eq!(
            ast,
            ASTNode::Sequence(vec![ASTNode::Return(Box::new(Expr::Mul(
                Box::new(Expr::Add(
                    Box::new(Expr::Number(2)),
                    Box::new(Expr::Number(3))
                )),
                Box::new(Expr::Number(4))
            )))]),
        );
    }

    #[test]
    fn test_if_else_parsing() {
        let src = "int main() { if (3 < 4) { return 1; } else { return 0; } }";
        let tokens = lexify_source(src);
        let ast = build_ast(&tokens);
        assert_eq!(
            ast,
            ASTNode::Sequence(vec![ASTNode::If {
                condition: Box::new(Expr::Less(
                    Box::new(Expr::Number(3)),
                    Box::new(Expr::Number(4))
                )),
                then_branch: Box::new(ASTNode::Sequence(vec![ASTNode::Return(Box::new(
                    Expr::Number(1)
                ))])),
                else_branch: Some(Box::new(ASTNode::Sequence(vec![ASTNode::Return(
                    Box::new(Expr::Number(0))
                )])))
            }]),
        );
    }

    #[test]
    fn test_while_loop_parsing() {
        let src = "int main() { while (0 < 1) { return 10; } }";
        let tokens = lexify_source(src);
        let ast = build_ast(&tokens);
        assert_eq!(
            ast,
            ASTNode::Sequence(vec![ASTNode::While {
                condition: Box::new(Expr::Less(
                    Box::new(Expr::Number(0)),
                    Box::new(Expr::Number(1))
                )),
                body: Box::new(ASTNode::Sequence(vec![ASTNode::Return(Box::new(
                    Expr::Number(10)
                ))]))
            }])
        );
    }

    #[test]
    fn test_variable_assignment_parsing() {
        let tokens = lexify_source("int x = 10; if (x == 10) { return x; }");
        let expected = vec![
            Token::Int,
            Token::Identifier("x".to_string()),
            Token::Assign,
            Token::Number(10),
            Token::Semicolon,
            Token::If,
            Token::LParen,
            Token::Identifier("x".to_string()),
            Token::Equal,
            Token::Number(10),
            Token::RParen,
            Token::LBrace,
            Token::Return,
            Token::Identifier("x".to_string()),
            Token::Semicolon,
            Token::RBrace,
        ];
        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_variable_return_function() {
        let tokens = lexify_source("int main() { int y = 7; return y; }");
        let ast = build_ast(&tokens);
        let (instructions, data_segment) = generate_bytecode(&ast);
        let mut vm = VM::new(instructions, data_segment);
        vm.execute_bytecode();
        assert_eq!(vm.stack.last(), Some(&7));
    }

    #[test]
    fn test_function_call_codegen() {
        let ast = ASTNode::Sequence(vec![
            ASTNode::FunctionDef {
                name: "mul".into(),
                params: vec!["x".into(), "y".into()],
                body: Box::new(ASTNode::Return(Box::new(Expr::Mul(
                    Box::new(Expr::Variable("x".into())),
                    Box::new(Expr::Variable("y".into())),
                )))),
            },
            ASTNode::Return(Box::new(Expr::Call(
                "mul".into(),
                vec![Expr::Number(3), Expr::Number(4)],
            ))),
        ]);
        let (instructions, _data_segment) = generate_bytecode(&ast);
        assert_eq!(
            instructions,
            vec![
                Instruction::ENT(2),
                Instruction::LEA(0),
                Instruction::LI,
                Instruction::LEA(1),
                Instruction::LI,
                Instruction::MUL,
                Instruction::PSH,
                Instruction::EXIT,
                Instruction::IMM(3),
                Instruction::IMM(4),
                Instruction::JSR(0),
                Instruction::PSH,
                Instruction::EXIT,
            ]
        );
    }

    #[test]
    fn test_empty_string_tokenization() {
        let tokens = lexify_source("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_logical_expression_parsing() {
        let tokens = lexify_source("int main() { return 1 < 2 == 1; }");
        let ast = build_ast(&tokens);
        match ast {
            ASTNode::Sequence(v) => match &v[0] {
                ASTNode::Return(expr) => match &**expr {
                    Expr::Equal(left, right) => {
                        matches!(**left, Expr::Less(_, _));
                        matches!(**right, Expr::Number(1));
                    }
                    _ => panic!("Expected equality expression"),
                },
                _ => panic!("Expected return node"),
            },
            _ => panic!("Expected sequence node"),
        }
    }

    #[test]
    fn test_syscall_behavior() {
        let program = vec![
            Instruction::IMM(0), // Address of string in data segment
            Instruction::IMM(0), // No format arguments
            Instruction::PRTF,  // Print string
            Instruction::MALC,  // Allocate memory
            Instruction::IMM(3),
            Instruction::CLOS,  // Close file
            Instruction::EXIT,
        ];
        let data_segment = vec!["test\n".to_string()];
        let mut vm = VM::new(program, data_segment);
        vm.execute_bytecode();
        assert_eq!(vm.stack.len(), 3);
        assert_eq!(vm.stack, vec![5, 0x1000, 0]);
    }

    #[test]
    fn test_string_literal_tokenization() {
        let src = "int main() { printf(\"hello world\\n\"); return 0; }";
        let tokens = lexify_source(src);
        assert_eq!(
            tokens,
            vec![
                Token::Int,
                Token::Identifier("main".into()),
                Token::LParen,
                Token::RParen,
                Token::LBrace,
                Token::Identifier("printf".into()),
                Token::LParen,
                Token::StringLiteral("hello world\n".into()),
                Token::RParen,
                Token::Semicolon,
                Token::Return,
                Token::Number(0),
                Token::Semicolon,
                Token::RBrace,
            ]
        );
    }
}
