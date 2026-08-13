#+feature using-stmt
// SPDX-FileCopyrightText: © 2025 Viktor M. <dayllenger@gmail.com>
// SPDX-License-Identifier: Zlib
//
// The tokenizer follows CSS Syntax 3 document: https://www.w3.org/TR/css-syntax-3.
// Tokenization has little to no comments since that document is very descriptive.
// Parsing order was slightly changed. There's no EOF token.
//
// The tokenizer is fast and outputs ~50M tokens per second.
package polyraster_css

import "base:runtime"

Token :: struct {
	kind:          Token_Kind,
	flags:         Token_Flags,
	delim:         rune,
	text:          string, // points inside the scratch buffer
	number:        f64,
	unit_offset:   i32, // offset inside `text`
	unicode_range: [2]rune,
	line:          i32, // starting at 1
	column:        i32, // starting at 1
}
#assert(size_of(Token) <= 64)

Token_Kind :: enum u8 {
	Delim, // delimiter - a single code point, may be an error
	Ident, // identifier
	Func, // function( - includes parenthesis
	At_Keyword, // @keyword - text will contain keyword w/o @ prefix
	Hash, // #
	Str, // string in '' or ""
	Bad_Str, // string ended with newline character
	Url, // url(), including url('string')
	Bad_Url, // bad url()
	Number, // +12345.321e-3
	Percentage, // 120%
	Dimension, // 1.23px - number with unit
	Unicode_Range, // U+XXX-XXX
	Match_Dash, // |=
	Match_Word, // ~=
	Match_Prefix, // ^=
	Match_Suffix, // $=
	Match_Substring, // *=
	Whitespace, // space, tab, newline, or a comment
	Cdo, // <!--
	Cdc, // -->
	Colon, // :
	Semicolon, // ;
	Comma, // ,
	Open_Paren, // (
	Close_Paren, // )
	Open_Square, // [
	Close_Square, // ]
	Open_Curly, // {
	Close_Curly, // }
}

Token_Flag :: enum u8 {
	Id,
	Integer,
}
Token_Flags :: bit_set[Token_Flag]

Tokenizer :: struct {
	s:           []byte, // scratch buffer
	i:           int, // current offset in the source
	line:        int, // current line
	line_offset: int,
	tokens:      [dynamic]Token,
}

Option :: enum u8 {
	Skip_Whitespace, // Do not output Whitespace tokens.
}
Options :: bit_set[Option]

// Parse the whole string into tokens. This procedure cannot fail. It returns
// the tokenizer that contains a scratch buffer and the token array, both
// allocated with the provided allocator. Token's `.text` points to the scratch
// buffer.
//
// Invalid escaped code points and unicode ranges as well as 0x0 are set to
// 0xFF, which itself is invalid and will be decoded as 0xFFFD when you for-loop
// over the text.
tokenize_string :: proc(src: string, options := Options{}, allocator: runtime.Allocator) -> Tokenizer {
	tzr := Tokenizer {
		s      = make([]byte, len(src) + 4, allocator),
		line   = 1,
		tokens = make([dynamic]Token, allocator),
	}
	copy(tzr.s, src)

	for {
		line := tzr.line
		column := tzr.i - tzr.line_offset + 1
		t: Token = consume_token(&tzr)
		if t.kind == .Delim && t.delim == 0 {
			break
		}
		t.line = i32(line)
		t.column = i32(column)
		if .Skip_Whitespace not_in options || t.kind != .Whitespace {
			append(&tzr.tokens, t)
		}
	}

	return tzr
}

/*
To filter code points from a stream of (unfiltered) code points input:
- Replace any CR code point, FF code point, or pairs of CR-LF by a single LF code point.
- Replace any NUL code point with U+FFFD REPLACEMENT CHARACTER.

(not used anymore but kept as a reference)
*/
preprocess_input :: proc(src: string, pad := true) -> []rune {
	res := make([]rune, len(src) + 4)
	count := 0
	was_cr: bool

	for c in src {
		if c == 0 {
			res[count] = 0xFFFD
			count += 1
		} else if c == '\r' || c == '\f' {
			res[count] = '\n'
			count += 1
		} else if c != '\n' || !was_cr {
			res[count] = c
			count += 1
		}
		was_cr = (c == '\r')
	}
	// pad with enough zeros to not worry about bounds
	return pad ? res : res[:count]
}

// ---

is_alpha :: proc "contextless" (c: byte) -> bool {
	return ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z')
}

is_digit :: proc "contextless" (c: byte) -> bool {
	return '0' <= c && c <= '9'
}

is_hex_digit :: proc "contextless" (c: byte) -> bool {
	return ('0' <= c && c <= '9') || ('A' <= c && c <= 'F') || ('a' <= c && c <= 'f')
}

is_white_space :: proc "contextless" (c: byte) -> bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\f' || c == '\r'
}

is_newline :: proc "contextless" (c1, c2: byte) -> bool {
	return c1 == '\n' || c1 == '\f' || (c1 == '\r' && c2 != '\n')
}

is_name_start :: proc "contextless" (c: byte) -> bool {
	return is_alpha(c) || c >= 0x80 || c == '_'
}

is_name :: proc "contextless" (c: byte) -> bool {
	return is_name_start(c) || is_digit(c) || c == '-'
}

is_non_printable :: proc "contextless" (c: byte) -> bool {
	return 0 <= c && c <= 0x08 || 0x0E <= c && c <= 0x1F || c == 0x0B || c == 0x7F
}

@(private)
is_eof :: proc "contextless" (c: byte) -> bool {
	return c == 0
}

starts_valid_escape :: proc "contextless" (c1, c2: byte) -> bool {
	return c1 == '\\' && !(c2 == '\n' || c2 == '\f' || c2 == '\r')
}

starts_with_ident :: proc "contextless" (s: []byte) -> bool {
	if s[0] == '-' {
		return is_name_start(s[1]) || s[1] == '-' || starts_valid_escape(s[1], s[2])
	}
	if is_name_start(s[0]) {
		return true
	}
	if s[0] == '\\' {
		return starts_valid_escape(s[0], s[1])
	}
	return false
}

starts_with_number :: proc "contextless" (s: []byte) -> bool {
	if s[0] == '+' || s[0] == '-' {
		return is_digit(s[1]) || s[1] == '.' && is_digit(s[2])
	}
	if s[0] == '.' {
		return is_digit(s[1])
	}
	return is_digit(s[0])
}

parse_hex_digit :: proc "contextless" (c: byte) -> u32 {
	if '0' <= c && c <= '9' {return u32(c - '0')}
	if 'a' <= c && c <= 'f' {return u32(c - 'a' + 10)}
	if 'A' <= c && c <= 'F' {return u32(c - 'A' + 10)}
	return ~u32(0)
}

@(private)
is_surrogate :: proc "contextless" (c: rune) -> bool {
	return 0xD800 <= c && c < 0xE000
}

// ---

consume_token :: proc(using tzr: ^Tokenizer) -> Token {
	c := s[i]
	i += 1

	if is_white_space(c) {
		i -= 1
		consume_white_space(tzr)
		return Token{kind = .Whitespace}
	}
	if c == '"' || c == '\'' {
		return consume_string(tzr, c)
	}
	if c == '#' {
		if is_name(s[i]) || starts_valid_escape(s[i], s[i + 1]) {
			t := Token {
				kind = .Hash,
			}
			if starts_with_ident(s[i:]) {
				t.flags |= {.Id}
			}
			t.text = consume_name(tzr)
			return t
		} else {
			return delim(s[i - 1:])
		}
	}
	if c == '+' {
		if starts_with_number(s[i - 1:]) {
			i -= 1
			return consume_numeric(tzr)
		} else {
			return delim(s[i - 1:])
		}
	}
	if c == '-' {
		if starts_with_number(s[i - 1:]) {
			i -= 1
			return consume_numeric(tzr)
		} else if starts_with_ident(s[i - 1:]) {
			i -= 1
			return consume_ident_like(tzr)
		} else if s[i] == '-' && s[i + 1] == '>' {
			i += 2
			return Token{kind = .Cdc}
		} else {
			return delim(s[i - 1:])
		}
	}
	if c == '.' {
		if starts_with_number(s[i - 1:]) {
			i -= 1
			return consume_numeric(tzr)
		} else {
			return delim(s[i - 1:])
		}
	}
	if c == '/' {
		if s[i] == '*' {
			i += 1
			start := i
			for !(s[i] == '*' && s[i + 1] == '/') && !is_eof(s[i + 2]) {
				if is_newline(s[i], s[i + 1]) {
					line += 1
					line_offset = i
				}
				i += 1
			}
			i += 2
			return Token{kind = .Whitespace, text = string(s[start:i - 2])}
		} else {
			return delim(s[i - 1:])
		}
	}
	if c == '<' {
		if s[i] == '!' && s[i + 1] == '-' && s[i + 2] == '-' {
			i += 3
			return Token{kind = .Cdo}
		} else {
			return delim(s[i - 1:])
		}
	}
	if c == '@' {
		if starts_with_ident(s[i:]) {
			return Token{kind = .At_Keyword, text = consume_name(tzr)}
		} else {
			return delim(s[i - 1:])
		}
	}
	if c == '\\' {
		if starts_valid_escape(c, s[i]) {
			i -= 1
			return consume_ident_like(tzr)
		} else {
			return delim(s[i - 1:])
		}
	}
	if c == '~' {
		if s[i] == '=' {
			i += 1
		} else {
			return delim(s[i - 1:])
		}
		return Token{kind = .Match_Word}
	}
	if c == '|' {
		if s[i] == '=' {
			i += 1
			return Token{kind = .Match_Dash}
		} else {
			return delim(s[i - 1:])
		}
	}
	if c == '^' {
		if s[i] == '=' {
			i += 1
			return Token{kind = .Match_Prefix}
		} else {
			return delim(s[i - 1:])
		}
	}
	if c == '$' {
		if s[i] == '=' {
			i += 1
			return Token{kind = .Match_Suffix}
		} else {
			return delim(s[i - 1:])
		}
	}
	if c == '*' {
		if s[i] == '=' {
			i += 1
			return Token{kind = .Match_Substring}
		} else {
			return delim(s[i - 1:])
		}
	}
	if c == ',' {return Token{kind = .Comma}}
	if c == ':' {return Token{kind = .Colon}}
	if c == ';' {return Token{kind = .Semicolon}}
	if c == '(' {return Token{kind = .Open_Paren}}
	if c == ')' {return Token{kind = .Close_Paren}}
	if c == '[' {return Token{kind = .Open_Square}}
	if c == ']' {return Token{kind = .Close_Square}}
	if c == '{' {return Token{kind = .Open_Curly}}
	if c == '}' {return Token{kind = .Close_Curly}}
	if is_digit(c) {
		i -= 1
		return consume_numeric(tzr)
	}
	if c == 'U' || c == 'u' {
		if s[i] == '+' && (is_hex_digit(s[i + 1]) || s[i + 1] == '?') {
			i += 1
			return consume_unicode_range(tzr)
		} else {
			i -= 1
			return consume_ident_like(tzr)
		}
	}
	if is_name_start(c) {
		i -= 1
		return consume_ident_like(tzr)
	}
	if is_eof(c) {
		return Token{}
	}
	return delim(s[i - 1:])
}

@(private)
delim :: proc(s: []byte) -> Token {
	for r in string(s) {
		return Token{kind = .Delim, delim = r != 0 ? r : 0xFFFD}
	}
	return Token{kind = .Delim} // unreachable
}

@(private)
consume_white_space :: proc(using tzr: ^Tokenizer) {
	for ; is_white_space(s[i]); i += 1 {
		if is_newline(s[i], s[i + 1]) {
			line += 1
			line_offset = i
		}
	}
}

@(private)
consume_string :: proc(using tzr: ^Tokenizer, quote: byte) -> Token {
	start, curr := i, i
	for {
		c := s[i]
		i += 1
		if is_eof(c) || c == quote {
			return Token{kind = .Str, text = string(s[start:curr])}
		}
		if c == '\n' || c == '\f' || c == '\r' {
			i -= 1
			return Token{kind = .Bad_Str, text = string(s[start:curr])}
		}
		if c == '\\' {
			if is_eof(s[i]) {
				continue
			}
			if is_newline(s[i], s[i + 1]) {
				line += 1
				if s[i] == '\r' && s[i + 1] == '\n' {
					i += 1
				}
				line_offset = i
				i += 1
			} else if starts_valid_escape(c, s[i]) {
				b, w := consume_escaped(tzr)
				for c in b[:w] {
					s[curr] = c
					curr += 1
				}
			}
		} else {
			s[curr] = c
			curr += 1
		}
	}
}

@(private)
consume_name :: proc(using tzr: ^Tokenizer) -> string {
	start, curr := i, i
	for {
		c := s[i]
		i += 1
		if is_name(c) {
			s[curr] = c
			curr += 1
		} else if starts_valid_escape(c, s[i]) {
			b, w := consume_escaped(tzr)
			for c in b[:w] {
				s[curr] = c
				curr += 1
			}
		} else {
			i -= 1
			return string(s[start:curr])
		}
	}
}

@(private)
consume_escaped :: proc(using tzr: ^Tokenizer) -> ([4]u8, int) {
	c := s[i]
	i += 1
	if is_hex_digit(c) {
		hex: u32 = parse_hex_digit(c)
		for j := 0; j < 5 && is_hex_digit(s[i]); j += 1 {
			hex <<= 4
			hex |= parse_hex_digit(s[i])
			i += 1
		}
		if is_white_space(s[i]) {
			i += 1
		}
		if hex == 0 || is_surrogate(rune(hex)) || hex > 0x10FFFF {
			return 0xFF, 1
		}
		return encode_rune(rune(hex))
	} else if is_eof(c) {
		return 0xFF, 1
	} else {
		return c, 1
	}
}

@(private)
consume_numeric :: proc(using tzr: ^Tokenizer) -> Token {
	start := i
	number, is_integer := consume_number(tzr)
	curr := i
	t := Token {
		kind   = .Number,
		flags  = is_integer ? {.Integer} : {},
		number = number,
	}
	if starts_with_ident(s[i:]) {
		t.kind = .Dimension
		t.unit_offset = i32(i - start)
		curr += len(consume_name(tzr))
	} else if s[i] == '%' {
		t.kind = .Percentage
		t.unit_offset = i32(i - start)
		curr += 1
		i += 1
	}
	t.text = string(s[start:curr])
	return t
}

@(private)
consume_number :: proc(using tzr: ^Tokenizer) -> (f64, bool) {
	is_integer := true
	num, frac, frac_exp: i64 = 0, 0, 1
	sign, exp, exp_sign: i64 = 1, 0, 1
	result: f64

	//odinfmt: disable
	@(static, rodata) pow10 := [?]f64{
		1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,
		1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
		1e20, 1e21, 1e22,
	}
	//odinfmt: enable

	if s[i] == '+' || s[i] == '-' {
		if s[i] == '-' {
			sign = -1
		}
		i += 1
	}
	for is_digit(s[i]) {
		num = num * 10 + i64(s[i] - '0')
		i += 1
	}
	if s[i] == '.' && is_digit(s[i + 1]) {
		i += 1
		for is_digit(s[i]) {
			frac = frac * 10 + i64(s[i] - '0')
			frac_exp *= 10
			i += 1
		}
		is_integer = false
	}
	if s[i] == 'e' || s[i] == 'E' {
		if (is_digit(s[i + 1]) || s[i + 1] == '-' || s[i + 1] == '+') && is_digit(s[i + 2]) {
			i += 1
			if s[i] == '-' {
				exp_sign = -1
				i += 1
			} else if s[i] == '+' {
				i += 1
			}
			for is_digit(s[i]) {
				exp = exp * 10 + i64(s[i] - '0')
				i += 1
			}
			is_integer = false
		}
	}

	result = f64(num) + f64(frac) / f64(frac_exp)
	if sign < 0 {
		result = -result
	}
	if exp < len(pow10) {
		result *= (exp_sign > 0 ? pow10[exp] : 1 / pow10[exp])
	} else {
		result *= (exp_sign > 0 ? 0h7ff00000_00000000 : 0)
	}

	return result, is_integer
}

@(private)
is_url_keyword :: proc(name: string) -> bool {
	if len(name) != 3 {
		return false
	}
	c0, c1, c2 := name[0], name[1], name[2]
	return (c0 == 'u' || c0 == 'U') && (c1 == 'r' || c1 == 'R') && (c2 == 'l' || c2 == 'L')
}

@(private)
consume_ident_like :: proc(using tzr: ^Tokenizer) -> Token {
	name := consume_name(tzr)
	if is_url_keyword(name) && s[i] == '(' {
		i += 1
		return consume_url(tzr)
	} else if s[i] == '(' {
		i += 1
		return Token{kind = .Func, text = name}
	} else {
		return Token{kind = .Ident, text = name}
	}
}

@(private)
consume_url :: proc(using tzr: ^Tokenizer) -> (t: Token) {
	t.kind = .Url
	consume_white_space(tzr)
	if is_eof(s[i]) {
		return
	}
	start, curr := i, i
	if s[i] == '"' || s[i] == '\'' {
		ending := s[i]
		i += 1
		stok := consume_string(tzr, ending)
		if stok.kind == .Bad_Str {
			return bad_url(tzr, start)
		}
		t.text = stok.text
		consume_white_space(tzr)
		if s[i] == ')' || is_eof(s[i]) {
			i += 1
			return
		} else {
			return bad_url(tzr, start)
		}
	}
	for {
		c := s[i]
		i += 1
		if c == ')' || is_eof(c) {
			t.text = string(s[start:curr])
			return
		}
		if is_white_space(c) {
			consume_white_space(tzr)
			if s[i] == ')' || is_eof(s[i]) {
				i += 1
				t.text = string(s[start:curr])
				return
			} else {
				return bad_url(tzr, start)
			}
		}
		if c == '"' || c == '\'' || c == '(' || is_non_printable(c) {
			return bad_url(tzr, start)
		}
		if c == '\\' {
			if starts_valid_escape(c, s[i]) {
				b, w := consume_escaped(tzr)
				for c in b[:w] {
					s[curr] = c
					curr += 1
				}
			} else {
				return bad_url(tzr, start)
			}
		} else {
			s[curr] = c
			curr += 1
		}
	}
}

@(private)
bad_url :: proc(using tzr: ^Tokenizer, start: int) -> Token {
	for {
		c := s[i]
		i += 1
		if c == ')' || is_eof(c) {
			break
		}
		if starts_valid_escape(c, s[i + 1]) {
			consume_escaped(tzr)
		}
	}
	return Token{kind = .Bad_Url, text = string(s[start:i - 1])}
}

@(private)
consume_unicode_range :: proc(using tzr: ^Tokenizer) -> Token {
	hex: [6]byte
	j: int
	question_marks: bool

	for ; j < 6 && is_hex_digit(s[i]); j += 1 {
		hex[j] = s[i]
		i += 1
	}
	for ; j < 6 && s[i] == '?'; j += 1 {
		hex[j] = '?'
		question_marks = true
		i += 1
	}
	for ; j < 6; j += 1 {
		hex[j] = 0
	}

	start: u32
	end: u32
	if question_marks {
		for k := 0; k < 6 && hex[k] != 0; k += 1 {
			start <<= 4
			end <<= 4
			start |= hex[k] == '?' ? 0x0 : parse_hex_digit(hex[k])
			end |= hex[k] == '?' ? 0xF : parse_hex_digit(hex[k])
		}
		return Token{kind = .Unicode_Range, unicode_range = {rune(start), rune(end)}}
	} else {
		for k := 0; k < 6 && hex[k] != 0; k += 1 {
			start <<= 4
			start |= parse_hex_digit(hex[k])
		}
	}
	if s[i] == '-' && is_hex_digit(s[i + 1]) {
		i += 1
		for k := 0; k < 6 && is_hex_digit(s[i]); k += 1 {
			end <<= 4
			end |= parse_hex_digit(hex[k])
			i += 1
		}
	} else {
		end = start
	}
	return Token{kind = .Unicode_Range, unicode_range = {rune(start), rune(end)}}
}

// extracted from "core:unicode/utf8"
@(private)
encode_rune :: proc "contextless" (r: rune) -> ([4]u8, int) {
	buf: [4]u8
	i := u32(r)
	mask :: u8(0x3f)
	if i <= 1 << 7 - 1 {
		buf[0] = u8(r)
		return buf, 1
	}
	if i <= 1 << 11 - 1 {
		buf[0] = 0xc0 | u8(r >> 6)
		buf[1] = 0x80 | u8(r) & mask
		return buf, 2
	}
	if i <= 1 << 16 - 1 {
		buf[0] = 0xe0 | u8(r >> 12)
		buf[1] = 0x80 | u8(r >> 6) & mask
		buf[2] = 0x80 | u8(r) & mask
		return buf, 3
	}
	buf[0] = 0xf0 | u8(r >> 18)
	buf[1] = 0x80 | u8(r >> 12) & mask
	buf[2] = 0x80 | u8(r >> 6) & mask
	buf[3] = 0x80 | u8(r) & mask
	return buf, 4
}
