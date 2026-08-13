#+feature using-stmt
// SPDX-FileCopyrightText: © 2025 Viktor M. <dayllenger@gmail.com>
// SPDX-License-Identifier: Zlib
//
// This is an abstract parser of CSS. Selectors are parsed enough to match
// them with elements, but properties and especially @at-rules contain
// generic token arrays.
//
// XML namespaces (|) are not supported.
package polyraster_css

import "base:runtime"

// Style sheet is a container of at-rules and qualified rules
Style_Sheet :: struct {
	at_rules: []At_Rule,
	rules:    []Rule,
}

// At-rule, like `@keyword prelude;` or `@keyword prelude { contents }`
At_Rule :: struct {
	keyword:  string,
	prelude:  []Token,
	contents: []Token,
}

// Qualified rule - selector list (comma-separated) and set of style properties
Rule :: struct {
	selectors:       []Selector,
	properties:      []Property,
	nested_rules:    []Rule,
	nested_at_rules: []At_Rule,
}

// Style property with a name and a list of tokens containing its value
Property :: struct {
	name:  string,
	value: []Token,
}

// Complex selector - one selector from the selector list
Selector :: struct {
	using last:  Selector_Part,
	/*
		Selector specificity:
		0 - the number of ID selectors,
		1 - the number of class, attribute, pseudo-class selectors,
		3 - the number of tag selectors and pseudo-elements.
	*/
	specificity: [3]u32,
	// Line of the first token
	line:        i32,
}

// Compound selector - a parsed sequence of simple selectors
Selector_Part :: struct {
	// Linked list of selector parts. Selectors in nested rules or from :is()
	// will point to the one shared selector prefix. Selectors of nested rules
	// are rewritten using :is().
	previous:        ^Selector_Part,
	// Element type or a custom name
	tag:             string,
	// ID, `#id`
	id:              string,
	// List of `.class` names
	classes:         []string,
	// List of `[attr]` variations
	attributes:      []Selector_Attr,
	// Filled in cases where there's just one :not() function with one selector inside
	not_function:    ^Selector_Part,
	// Functional pseudo-classes or pseudo-elements, e.g. `:nth-child(...)` or `::part(...)`.
	functions:       []Selector_Function,
	// A generic bit set of state-related `:pseudo-class` items
	pseudo_classes:  u64,
	// `::pseudo-element`
	pseudo_element:  string,
	// Tree-Structural pseudo-classes, e.g. `:root` or `:first-child`
	tree_structural: bit_set[Tree_Structural_Pseudo_Class],
	// When false, it should be a universal '*' selector without id, tag, etc.
	has_something:   bool,
	// Combinator between this selector and the `previous`
	combinator:      Selector_Combinator,
}

Selector_Function :: struct {
	func:      enum u32 {
		Other, // no selector inside
		Is, // :is(), :where()
		Not, // :not(s1, s2, ...)
		Has, // :has()
		Nth, // :nth-*(... of <selector>)
		Host, // :host()
	},
	a, b:      i16, // An+B syntax
	selectors: []^Selector_Part,
	name:      string,
	args:      []Token,
}

Selector_Attr :: struct {
	name:     string,
	value:    string,
	pattern:  enum u32 {
		Invalid,
		Any, // [attr]
		Exact, // [attr=value]
		Dash, // [attr|=value]
		Word, // [attr~=value]
		Prefix, // [attr^=value]
		Suffix, // [attr$=value]
		Substring, // [attr*=value]
	},
	modifier: rune, // case sensitivity: i, s
}

Tree_Structural_Pseudo_Class :: enum u8 {
	Root,
	Empty,
	First_Child,
	Last_Child,
	Only_Child,
	First_Of_Type,
	Last_Of_Type,
	Only_Of_Type,
}

Selector_Combinator :: enum u8 {
	Descendant, // ' '
	Child, // >
	Next, // +
	Subsequent, // ~
}

Parser_Options :: struct {
	pseudo_class_handler: proc(ident: string) -> u64,
	error_handler:        proc(tok: Token, msg: string),
	keep_empty_rules:     bool,
}

// Parse the whole style sheet.
//
// `pseudo_class_handler` is a user-provided callback that translates names such as "visited" into a bit flag.
parse :: proc(tokens: []Token, options: Parser_Options, allocator: runtime.Allocator) -> Style_Sheet {
	p := Parser{tokens, 0, options}
	if p.opt.error_handler == nil {
		p.opt.error_handler = default_error_handler
	}
	context.allocator = allocator
	return consume_rules(&p)
}

// Parse inline style. Better tokenize with `Skip_Whitespace`.
parse_properties :: proc(tokens: []Token, options: Parser_Options, allocator: runtime.Allocator) -> []Property {
	p := Parser{tokens, 0, options}
	if p.opt.error_handler == nil {
		p.opt.error_handler = default_error_handler
	}
	context.allocator = allocator
	properties := make([dynamic]Property)
	consume_declaration_list(&p, &properties)
	return properties[:]
}

// If some of selectors are invalid, the function will return the list of correct ones,
// but `ok` will be set to false. This allows for "forgiving" selector parsing.
// Note: you don't want to tokenize with `Skip_Whitespace` since descendant combinator is a whitespace.
parse_selectors :: proc(
	tokens: []Token,
	options: Parser_Options,
	allocator: runtime.Allocator,
) -> (
	list: []Selector,
	ok: bool,
) {
	p := Parser{tokens, 0, options}
	if p.opt.error_handler == nil {
		p.opt.error_handler = default_error_handler
	}
	context.allocator = allocator
	return consume_selector_list(&p, nil)
}

default_error_handler :: proc(tok: Token, msg: string) {
	runtime.print_string("CSS(")
	runtime.print_i64(i64(tok.line))
	runtime.print_string(":")
	runtime.print_i64(i64(tok.column))
	runtime.print_strings("): ", msg, "\n")
}

// True if a < b.
less_specific :: proc "contextless" (a, b: [3]u32) -> bool {
	if a[0] != b[0] {
		return a[0] < b[0]
	}
	if a[1] != b[1] {
		return a[1] < b[1]
	}
	return a[2] < b[2]
}

@(private)
Parser :: struct {
	r:   []Token, // range
	i:   int,
	opt: Parser_Options,
}

@(private)
consume_rules :: proc(using p: ^Parser) -> Style_Sheet {
	at_rules: [dynamic]At_Rule
	rules: [dynamic]Rule

	for i < len(r) {
		if r[i].kind == .Whitespace || r[i].kind == .Cdo || r[i].kind == .Cdc {
			i += 1
			continue
		}
		if r[i].kind == .At_Keyword {
			if at_r, ok := consume_at_rule(p); ok {
				append(&at_rules, at_r)
			}
		} else {
			if rule, ok := consume_qualified_rule(p, nil); ok {
				append(&rules, rule)
			}
		}
	}

	return Style_Sheet{at_rules[:], rules[:]}
}

@(private)
consume_at_rule :: proc(using p: ^Parser) -> (At_Rule, bool) {
	first_tok := r[i]
	rule := At_Rule {
		keyword = first_tok.text,
	}
	i += 1
	for i < len(r) && r[i].kind == .Whitespace {
		i += 1
	}

	start, end := i, i
	for i < len(r) {
		t := r[i]
		if t.kind == .Whitespace {
			i += 1
			continue
		}
		i += 1

		if t.kind == .Open_Curly {
			open_curly := i
			braces := 1
			for i < len(r) && braces > 0 {
				t = r[i]
				if t.kind == .Open_Curly {
					braces += 1
				} else if t.kind == .Close_Curly {
					braces -= 1
				}
				i += 1
			}
			if braces != 0 {
				opt.error_handler(t, "unmatched braces")
			}
			if open_curly == i {
				opt.error_handler(t, "empty block")
			}
			rule.contents = r[open_curly:i]
			break
		}
		if t.kind == .Semicolon {
			break
		}
		end = i
	}

	rule.prelude = r[start:end]
	if len(rule.prelude) > 0 || len(rule.contents) > 0 {
		return rule, true
	}

	opt.error_handler(first_tok, "skipping empty rule")
	return {}, false
}

@(private)
consume_qualified_rule :: proc(using p: ^Parser, parent_selectors: []Selector) -> (rule: Rule, ok: bool) {
	properties: [dynamic]Property
	nested_rules: [dynamic]Rule
	nested_at_rules: [dynamic]At_Rule

	rule.selectors = consume_selector_list(p, parent_selectors) or_return
	if i < len(r) && r[i].kind == .Open_Curly {
		i += 1
	} else {
		return
	}

	for i < len(r) {
		t := r[i]
		if t.kind == .Whitespace || t.kind == .Semicolon {
			i += 1
			continue
		}
		if t.kind == .Close_Curly {
			i += 1
			break
		}
		if t.kind == .Ident && i + 1 < len(r) && r[i + 1].kind == .Colon {
			consume_declaration_list(p, &properties)
		} else if t.kind == .At_Keyword {
			if at_r, ok := consume_at_rule(p); ok {
				append(&nested_at_rules, at_r)
			}
		} else {
			if nested, ok := consume_qualified_rule(p, rule.selectors); ok {
				append(&nested_rules, nested)
			}
		}
	}

	if len(properties) > 0 || len(nested_rules) > 0 || opt.keep_empty_rules {
		rule.properties = properties[:]
		rule.nested_rules = nested_rules[:]
		rule.nested_at_rules = nested_at_rules[:]
		ok = true
	}
	return
}

@(private)
consume_selector_list :: proc(using p: ^Parser, parent_selectors: []Selector) -> ([]Selector, bool) {
	list: [dynamic]Selector

	for i < len(r) {
		if r[i].kind == .Open_Curly {
			break
		}
		if sel, ok := consume_selector(p, parent_selectors); ok {
			append(&list, sel)
		}
		if i < len(r) && r[i].kind == .Comma {
			i += 1
			for i < len(r) && r[i].kind == .Whitespace {
				i += 1
			}
		}
	}
	return list[:], true
}

@(private)
consume_selector :: proc(using p: ^Parser, parent_selectors: []Selector) -> (sel: Selector, ok: bool) {
	previous: ^Selector_Part
	is_first := true
	had_ampersand: bool

	for i < len(r) {
		// treat all selectors as relative, i.e. starting with a combinator
		combinator: Selector_Combinator
		if r[i].kind == .Whitespace || r[i].kind == .Delim {
			if combinator, ok = consume_combinator(p); ok && !is_first {
				previous = new_clone(sel.last)
			}
		}
		if i >= len(r) || r[i].kind == .Comma || r[i].kind == .Open_Curly {
			break
		}
		if sel.line == 0 {
			sel.line = r[i].line
		}

		sel.last = consume_selector_part(p, &had_ampersand, parent_selectors, &sel.specificity) or_return
		sel.last.has_something = runtime.memory_compare_zero(&sel.last, size_of(Selector_Part)) > 0

		sel.last.previous = previous
		sel.last.combinator = combinator
		is_first = false
	}

	// handle nested rules
	if len(parent_selectors) > 0 && !had_ampersand {
		the_first := &sel.last
		for the_first.previous != nil {
			the_first = the_first.previous
		}

		if len(parent_selectors) == 1 {
			the_first.previous = &parent_selectors[0].last
			sel.specificity += parent_selectors[0].specificity
		} else {
			// treat multiple parent selectors like :is(a, b, c)
			parent_is := new(Selector_Part)
			func := Selector_Function {
				func      = .Is,
				selectors = make([]^Selector_Part, len(parent_selectors)),
			}
			parent_is.has_something = true
			sfty: [3]u32
			for &parent, i in parent_selectors {
				func.selectors[i] = &parent_selectors[i].last
				if less_specific(sfty, parent_selectors[i].specificity) {
					sfty = parent_selectors[i].specificity
				}
			}
			parent_is.functions = make([]Selector_Function, 1)
			parent_is.functions[0] = func
			sel.specificity += sfty
			the_first.previous = parent_is
		}
	}

	return sel, true
}

@(private)
consume_selector_part :: proc(
	using p: ^Parser,
	had_ampersand: ^bool,
	parent_selectors: []Selector,
	specificity: ^[3]u32,
) -> (
	part: Selector_Part,
	ok: bool,
) {
	classes: [dynamic]string
	attributes: [dynamic]Selector_Attr
	functions: [dynamic]Selector_Function
	is_first := true

	for i < len(r) {
		t := r[i]
		if t.kind == .Whitespace || t.kind == .Comma || t.kind == .Open_Curly {
			break
		}

		if t.kind == .Delim && t.delim == '*' { 	// universal
			if !is_first {
				opt.error_handler(t, "tag and * must be first")
			}
			is_first = false
			i += 1
			continue
		} else if t.kind == .Ident { 	// tag
			if is_first {
				part.tag = t.text
				specificity[2] += 1
			} else {
				opt.error_handler(t, "tag and * must be first")
			}
			is_first = false
			i += 1
			continue
		} else {
			is_first = false
		}

		if t.kind == .Delim {
			if t.delim == '.' { 	// class
				i += 1
				if i < len(r) && r[i].kind == .Ident {
					append(&classes, r[i].text)
					specificity[1] += 1
					i += 1
				} else {
					opt.error_handler(t, "expected identifier as a class name")
				}

			} else if t.delim == '&' { 	// nesting
				i += 1
				had_ampersand^ = true
				// treat parent selectors like :is
				func := Selector_Function {
					func      = .Is,
					selectors = make([]^Selector_Part, len(parent_selectors)),
				}
				sfty: [3]u32
				for &parent, i in parent_selectors {
					func.selectors[i] = &parent.last
					if less_specific(sfty, parent.specificity) {
						sfty = parent.specificity
					}
				}
				append(&functions, func)
				specificity^ += sfty
				continue

			} else {
				// combinator or something
				if t.delim == '>' || t.delim == '+' || t.delim == '~' {
					break
				} else {
					opt.error_handler(t, "unexpected delimiter")
					i += 1
					return {}, false
				}
			}

		} else if t.kind == .Colon { 	// pseudo-classes and pseudo-elements
			i += 1
			if i >= len(r) {break}

			if r[i].kind == .Ident {
				switch r[i].text {
				case "root":
					part.tree_structural |= {.Root}
				case "empty":
					part.tree_structural |= {.Empty}
				case "first-child":
					part.tree_structural |= {.First_Child}
				case "last-child":
					part.tree_structural |= {.Last_Child}
				case "only-child":
					part.tree_structural |= {.Only_Child}
				case "first-of-type":
					part.tree_structural |= {.First_Of_Type}
				case "last-of-type":
					part.tree_structural |= {.Last_Of_Type}
				case "only-of-type":
					part.tree_structural |= {.Only_Of_Type}
				case:
					if opt.pseudo_class_handler != nil {
						part.pseudo_classes |= opt.pseudo_class_handler(r[i].text)
					}
				}
				specificity[1] += 1
				i += 1
				continue
			}

			if r[i].kind == .Colon {
				i += 1
				if i < len(r) && r[i].kind == .Ident {
					part.pseudo_element = r[i].text
					specificity[2] += 1
					i += 1
					continue
				} else if i < len(r) && r[i].kind == .Func {
					// functional pseudo-element
				} else {
					opt.error_handler(t, "expected identifier as a pseudo element")
					continue
				}
			}

			if r[i].kind == .Func {
				func := Selector_Function {
					name = r[i].text,
				}
				i += 1
				// get tokens
				t: Token
				start := i
				parens := 1
				for i < len(r) && parens > 0 {
					t = r[i]
					if t.kind == .Open_Paren || t.kind == .Func {
						parens += 1
					} else if t.kind == .Close_Paren {
						parens -= 1
					}
					i += 1
				}
				if parens != 0 {
					opt.error_handler(t, "unmatched parentheses")
				}
				if start >= i - 1 {
					opt.error_handler(t, "empty pseudo-class function")
					continue
				}
				inside_tokens := r[start:i - 1]

				switch func.name {
				case "not":
					func.func = .Not
				case "is", "where":
					func.func = .Is
				case "has":
					func.func = .Has
				case:
					if len(func.name) > 4 && func.name[:4] == "nth-" {
						for t, j in inside_tokens {
							if t.kind == .Ident && t.text == "of" {
								func.func = .Nth
								func.a, func.b = parse_an_plus_b(inside_tokens[:j])
								inside_tokens = inside_tokens[j + 1:]
								break
							}
						}
					}
				}

				if func.func != .Other {
					list, _ := parse_selectors(inside_tokens, opt, context.allocator)
					if len(list) == 0 {
						continue
					}
					if len(list) == 1 && func.func == .Not && part.not_function == nil {
						sel := list[0]
						if sel.pseudo_element == "" {
							part.not_function = new_clone(sel.last)
							specificity^ += sel.specificity
						}
						continue
					}

					func.selectors = make([]^Selector_Part, len(list))
					sfty: [3]u32
					n := 0
					for sel in list {
						if sel.pseudo_element == "" {
							if func.name != "where" {
								if less_specific(sfty, sel.specificity) {
									sfty = sel.specificity
								}
							}
							if func.func == .Nth {
								specificity[1] += 1
							}
							func.selectors[n] = new_clone(sel.last)
							n += 1
						}
					}
					func.selectors = func.selectors[:n]
					specificity^ += sfty

				} else if func.name == "host" {
					func.func = .Host
					specificity[1] += 1
					p2 := Parser{inside_tokens, 0, opt}
					sel := consume_selector(&p2, nil) or_return
					specificity^ += sel.specificity
					func.selectors = make([]^Selector_Part, 1)
					func.selectors[0] = new_clone(sel.last)

				} else {
					specificity[1] += 1
					func.args = inside_tokens
				}

				append(&functions, func)
				continue
			}

			opt.error_handler(t, "expected valid pseudo-class or pseudo-element in a selector")
			return

		} else if t.kind == .Hash && .Id in t.flags { 	// id
			if part.id == "" || part.id == t.text {
				part.id = t.text
				specificity[0] += 1
				i += 1
			} else {
				opt.error_handler(t, "two different #ids match nothing")
				return
			}

		} else if t.kind == .Open_Square { 	// attribute
			i += 1
			if a, ok := consume_attribute_selector(p); ok {
				append(&attributes, a)
				specificity[1] += 1
			}

		} else {
			opt.error_handler(t, "unexpected token in selector")
			i += 1
			return {}, false
		}
	}

	part.classes = classes[:]
	part.attributes = attributes[:]
	part.functions = functions[:]

	// check if attribute selectors can match something
	for &a in part.attributes {
		if len(a.value) > 0 {
			if a.pattern == .Word {
				for r in a.value {
					if is_white_space(byte(r)) {
						a.pattern = .Invalid
						break
					}
				}
			}
		} else {
			if a.pattern == .Word || a.pattern == .Prefix || a.pattern == .Suffix || a.pattern == .Substring {
				a.pattern = .Invalid
			}
		}
	}

	return part, true
}

parse_an_plus_b :: proc(tokens: []Token) -> (a, b: i16) {
	ts := tokens
	for len(ts) > 0 && ts[0].kind == .Whitespace {
		ts = ts[1:]
	}
	for len(ts) > 0 && ts[len(ts) - 1].kind == .Whitespace {
		ts = ts[:len(ts) - 1]
	}
	if len(ts) == 0 {
		return
	}

	parse_i16 :: proc(s: string) -> (i16, bool) {
		num: int
		for i in 0 ..< len(s) {
			if !is_digit(s[i]) {
				return 0, false
			}
			num = num * 10 + int(s[i] - '0')
			if num > int(max(i16)) {
				return 0, true
			}
		}
		return i16(num), true
	}

	range_check :: proc(f: f64) -> i16 {
		return f64(-max(i16)) <= f && f <= f64(max(i16)) ? i16(f) : 0
	}

	// this stuff is a little insane because the An+B grammar was not defined
	// in terms of tokens initially
	t0 := ts[0]
	if len(ts) == 1 {
		#partial switch t0.kind {
		case .Ident:
			if t0.text == "odd" {
				return 2, 1
			}
			if t0.text == "even" {
				return 2, 0
			}
			s := t0.text
			a = 1
			if s[0] == '-' {
				s = s[1:]
				a = -1
			}
			if s == "n" {
				return a, 0
			}
			if len(s) > 2 && s[:2] == "n-" {
				if b, ok := parse_i16(s[2:]); ok {
					return a, -b
				}
			}
		case .Number:
			if .Integer in t0.flags {
				return 0, range_check(t0.number)
			}
		case .Dimension:
			if .Integer in t0.flags {
				if t0.text[t0.unit_offset:] == "n" {
					return range_check(t0.number), 0
				}
			}
		}
		return 0, 0
	}

	if t0.kind == .Delim {
		if t0.delim == '+' { 	// no space is allowed here
			s := ts[1].text
			a = 1
			if s == "n" {
				ts = ts[2:]
				if len(ts) == 0 {
					return a, 0
				}
			} else if len(s) > 2 && s[:2] == "n-" {
				if b, ok := parse_i16(s[2:]); ok {
					return a, -b
				}
				return 0, 0
			} else {
				return 0, 0
			}
		} else {
			return 0, 0
		}
	} else if t0.kind == .Ident {
		s := t0.text
		a = 1
		if s[0] == '-' {
			s = s[1:]
			a = -1
		}
		ts = ts[1:]
		if s == "n-" {
			for len(ts) > 0 && ts[0].kind == .Whitespace {
				ts = ts[1:]
			}
			if len(ts) == 1 && ts[0].kind == .Number && .Integer in ts[0].flags {
				if is_digit(ts[0].text[0]) {
					return a, -range_check(ts[0].number)
				}
				return 0, 0
			}
		} else if s != "n" {
			return 0, 0
		}
	} else if t0.kind == .Dimension {
		if .Integer in t0.flags {
			a = range_check(t0.number)
			ts = ts[1:]
		} else {
			return 0, 0
		}
	} else {
		return 0, 0
	}

	for len(ts) > 0 && ts[0].kind == .Whitespace {
		ts = ts[1:]
	}
	if len(ts) > 0 && ts[0].kind == .Delim {
		if ts[0].delim == '-' {
			b = -1
		} else if ts[0].delim == '+' {
			b = 1
		} else {
			return 0, 0
		}
		ts = ts[1:]
	}
	for len(ts) > 0 && ts[0].kind == .Whitespace {
		ts = ts[1:]
	}
	if len(ts) == 1 && ts[0].kind == .Number && .Integer in ts[0].flags {
		if b != 0 {
			if is_digit(ts[0].text[0]) {
				return a, b * range_check(ts[0].number)
			}
		} else {
			if !is_digit(ts[0].text[0]) {
				return a, range_check(ts[0].number)
			}
		}
	}
	return 0, 0
}

@(private)
consume_attribute_selector :: proc(using p: ^Parser) -> (Selector_Attr, bool) {
	attr := Selector_Attr {
		pattern = .Any,
	}
	state := 0
	for i < len(r) {
		t := r[i]
		if t.kind == .Close_Square {
			i += 1
			break
		}
		if t.kind == .Whitespace {
			i += 1
			continue
		}
		if state == 0 {
			if t.kind == .Ident {
				attr.name = t.text
				state = 1
				i += 1
				continue
			} else {
				opt.error_handler(t, "expected identifier in attribute")
				return {}, false
			}
		}
		if state == 1 {
			if t.kind == .Delim && t.delim == '=' {
				attr.pattern = .Exact
			} else if t.kind == .Match_Dash {
				attr.pattern = .Dash
			} else if t.kind == .Match_Word {
				attr.pattern = .Word
			} else if t.kind == .Match_Prefix {
				attr.pattern = .Prefix
			} else if t.kind == .Match_Suffix {
				attr.pattern = .Suffix
			} else if t.kind == .Match_Substring {
				attr.pattern = .Substring
			} else {
				opt.error_handler(t, "unexpected token in attribute selector")
				return {}, false
			}
			state = 2
			i += 1
			continue
		}
		if state == 2 {
			if t.kind == .Ident || t.kind == .Str {
				if attr.name == "" {
				} else {
					attr.value = t.text
				}
				state = 3
				i += 1
				continue
			} else {
				opt.error_handler(t, "expected string, identifier, or close bracket in attribute")
				return {}, false
			}
		}
		if state == 3 {
			if t.kind == .Ident && len(t.text) == 1 {
				for r in t.text {
					attr.modifier = r
					break
				}
				i += 1
				continue
			}
		}
		opt.error_handler(t, "expected close bracket in attribute")
		break
	}
	return attr, true
}

@(private)
consume_combinator :: proc(using p: ^Parser) -> (comb: Selector_Combinator, ok: bool) {
	t: Token
	for i < len(r) {
		t := r[i]
		if t.kind == .Comma || t.kind == .Open_Curly {
			return {}, false
		}
		if t.kind == .Whitespace {
			i += 1
			ok = true
			continue
		}
		if t.kind == .Delim {
			if t.delim == '>' {
				i += 1
				comb = .Child
				ok = true
				continue
			}
			if t.delim == '+' {
				i += 1
				comb = .Next
				ok = true
				continue
			}
			if t.delim == '~' {
				i += 1
				comb = .Subsequent
				ok = true
				continue
			}
		}
		return
	}
	return
}

@(private)
consume_declaration_list :: proc(using p: ^Parser, list: ^[dynamic]Property) {
	for i < len(r) {
		t := r[i]
		if t.kind == .Whitespace {
			i += 1
			continue
		}
		if t.kind == .Ident {
			if prop, ok := consume_declaration(p); ok {
				append(list, prop)
			}
		} else {
			break
		}
	}
}

@(private)
consume_declaration :: proc(using p: ^Parser) -> (Property, bool) {
	prop := Property {
		name = r[i].text,
	}
	i += 1
	for i < len(r) && r[i].kind == .Whitespace {
		i += 1
	}
	if i >= len(r) || r[i].kind == .Close_Curly {
		return {}, false
	}

	t := r[i]
	if t.kind == .Colon {
		i += 1
		prop.value = consume_value(p)
		if len(prop.value) > 0 {
			return prop, true
		}
		// declaration can be empty in custom properties
		if len(prop.name) > 2 && prop.name[0] == '-' && prop.name[1] == '-' {
			return prop, true
		} else {
			opt.error_handler(t, "declaration is empty")
			return {}, false
		}
	}
	opt.error_handler(t, "expected colon in declaration")
	i += 1
	return {}, false
}

@(private)
consume_value :: proc(using p: ^Parser) -> []Token {
	for i < len(r) && r[i].kind == .Whitespace {
		i += 1
	}

	start, end := i, i
	ws := 0
	// we don't want whitespaces in values.
	// try to consume without allocating, fall back if needed
	for i < len(r) {
		t := r[i]
		if t.kind == .Close_Curly {
			break
		}
		i += 1
		if t.kind == .Semicolon {
			break
		}
		if t.kind == .Whitespace {
			ws += 1
		} else {
			end = i
		}
	}

	if ws > 0 {
		list := make([]Token, end - start)
		n := 0
		for t in r[start:end] {
			if t.kind != .Whitespace {
				list[n] = t
				n += 1
			}
		}
		return list[:n]
	}
	return r[start:end]
}
