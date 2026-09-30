package com.example;

import java.util.List;

/** Parses input. */
public class Parser {
    private final List<String> tokens;

    public Parser(List<String> tokens) {
        this.tokens = tokens;
    }

    @Override
    public String toString() {
        return "Parser";
    }
}

interface Visitor {
    void visit(Parser parser);
}
