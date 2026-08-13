#+test
package polyraster_css

import "core:fmt"
import "core:strings"
import "core:testing"

@(test)
test_tokenizer :: proc(_: ^testing.T) {
	src :=
		`
	identifier-1#id[*=12345] {
        'str1' "str2"
        -moz-what: 1.23px 0.75em   /* the comment */
        @keyword U+140?! -.234e+5 1.0e30;
        color:   #fe5 #000000
        --custom-3
        url(  'stuff.css')
        url(#grad1)
        url(bad url);
        url('apparently, \
good'
)

        url( ok )
        function(120%);
        '\30 \31'
        '\0 '
    }
    ` +
		"'str\\\ning''bad\n"

	tzr := tokenize_string(src, {}, context.temp_allocator)
	ts := tzr.tokens[:]

	next :: proc(ts: ^[]Token) -> Token {
		tok := ts[0]
		tok.line = 0
		tok.column = 0
		(ts^) = ts[1:]
		return tok
	}

	assert(next(&ts) == Token{kind = .Whitespace})
	assert(next(&ts) == Token{kind = .Ident, text = "identifier-1"})
	assert(next(&ts) == Token{kind = .Hash, flags = {.Id}, text = "id"})
	assert(next(&ts) == Token{kind = .Open_Square})
	assert(next(&ts) == Token{kind = .Match_Substring})
	assert(next(&ts) == Token{kind = .Number, flags = {.Integer}, text = "12345", number = 12345})
	assert(next(&ts) == Token{kind = .Close_Square})
	assert(next(&ts) == Token{kind = .Whitespace})
	assert(next(&ts) == Token{kind = .Open_Curly})
	assert(next(&ts) == Token{kind = .Whitespace})

	assert(next(&ts) == Token{kind = .Str, text = "str1"})
	assert(next(&ts) == Token{kind = .Whitespace})
	assert(next(&ts) == Token{kind = .Str, text = "str2"})
	assert(next(&ts) == Token{kind = .Whitespace})

	assert(next(&ts) == Token{kind = .Ident, text = "-moz-what"})
	assert(next(&ts) == Token{kind = .Colon})
	assert(next(&ts) == Token{kind = .Whitespace})
	assert(next(&ts) == Token{kind = .Dimension, text = "1.23px", number = 1.23, unit_offset = 4})
	assert(next(&ts) == Token{kind = .Whitespace})
	assert(next(&ts) == Token{kind = .Dimension, text = "0.75em", number = 0.75, unit_offset = 4})
	assert(next(&ts) == Token{kind = .Whitespace})
	assert(next(&ts) == Token{kind = .Whitespace, text = " the comment "})
	assert(next(&ts) == Token{kind = .Whitespace})

	assert(next(&ts) == Token{kind = .At_Keyword, text = "keyword"})
	assert(next(&ts) == Token{kind = .Whitespace})
	assert(next(&ts) == Token{kind = .Unicode_Range, unicode_range = {0x1400, 0x140F}})
	assert(next(&ts) == Token{kind = .Delim, delim = '!'})
	assert(next(&ts) == Token{kind = .Whitespace})
	assert(next(&ts) == Token{kind = .Number, text = "-.234e+5", number = -.234e+5})
	assert(next(&ts) == Token{kind = .Whitespace})
	assert(next(&ts) == Token{kind = .Number, text = "1.0e30", number = 0h7ff00000_00000000})
	assert(next(&ts) == Token{kind = .Semicolon})
	assert(next(&ts) == Token{kind = .Whitespace})

	assert(next(&ts) == Token{kind = .Ident, text = "color"})
	assert(next(&ts) == Token{kind = .Colon})
	assert(next(&ts) == Token{kind = .Whitespace})
	assert(next(&ts) == Token{kind = .Hash, flags = {.Id}, text = "fe5"})
	assert(next(&ts) == Token{kind = .Whitespace})
	assert(next(&ts) == Token{kind = .Hash, text = "000000"})
	assert(next(&ts) == Token{kind = .Whitespace})

	assert(next(&ts) == Token{kind = .Ident, text = "--custom-3"})
	assert(next(&ts) == Token{kind = .Whitespace})

	assert(next(&ts) == Token{kind = .Url, text = "stuff.css"})
	assert(next(&ts) == Token{kind = .Whitespace})

	assert(next(&ts) == Token{kind = .Url, text = "#grad1"})
	assert(next(&ts) == Token{kind = .Whitespace})

	assert(next(&ts) == Token{kind = .Bad_Url, text = "bad url"})
	assert(next(&ts) == Token{kind = .Semicolon})
	assert(next(&ts) == Token{kind = .Whitespace})

	assert(next(&ts) == Token{kind = .Url, text = "apparently, good"})
	assert(next(&ts) == Token{kind = .Whitespace})

	assert(next(&ts) == Token{kind = .Url, text = "ok"})
	assert(next(&ts) == Token{kind = .Whitespace})

	assert(next(&ts) == Token{kind = .Func, text = "function"})
	assert(next(&ts) == Token{kind = .Percentage, flags = {.Integer}, text = "120%", number = 120, unit_offset = 3})
	assert(next(&ts) == Token{kind = .Close_Paren})
	assert(next(&ts) == Token{kind = .Semicolon})
	assert(next(&ts) == Token{kind = .Whitespace})

	assert(next(&ts) == Token{kind = .Str, text = "01"})
	assert(next(&ts) == Token{kind = .Whitespace})
	assert(next(&ts) == Token{kind = .Str, text = "\xFF"})
	assert(next(&ts) == Token{kind = .Whitespace})

	assert(next(&ts) == Token{kind = .Close_Curly})
	assert(next(&ts) == Token{kind = .Whitespace})
	assert(next(&ts) == Token{kind = .Str, text = "string"})
	assert(next(&ts) == Token{kind = .Bad_Str, text = "bad"})
	assert(next(&ts) == Token{kind = .Whitespace})

	assert(len(ts) == 0)
	assert(tzr.line == 1 + strings.count(src, "\n"))
}

@(test)
test_parser_1 :: proc(_: ^testing.T) {
	src := `
    <!--
    @import url('second.css') ;

    nav.class p, second#id, :visited {
        color: #888;
        font-size: 120%;
    }
    -->
    `

	tzr := tokenize_string(src, {}, context.temp_allocator)
	style := parse(tzr.tokens[:], {}, context.temp_allocator)

	{
		rule := style.at_rules
		assert(len(rule) == 1)
		assert(rule[0].keyword == "import")
		assert(len(rule[0].prelude) == 1)
		assert(rule[0].prelude[0].kind == .Url)
		assert(rule[0].prelude[0].text == "second.css")
	}

	{
		sel: Selector
		part: ^Selector_Part

		assert(len(style.rules) == 1)
		rule := style.rules[0]
		assert(len(rule.selectors) == 3)

		sel = rule.selectors[0]
		assert(sel.specificity == [3]u32{0, 1, 2})
		assert(sel.tag == "p")
		assert(len(sel.classes) == 0)
		assert(sel.combinator == .Descendant)
		part = sel.previous
		assert(part != nil)
		assert(part.tag == "nav")
		assert(len(part.classes) == 1)
		assert(part.classes[0] == "class")
		assert(part.has_something)
		assert(part.previous == nil)

		sel = rule.selectors[1]
		assert(sel.specificity == [3]u32{1, 0, 1})
		assert(sel.tag == "second")
		assert(sel.id == "id")
		assert(len(sel.classes) == 0)
		assert(sel.has_something)
		assert(sel.previous == nil)

		sel = rule.selectors[2]
		assert(sel.specificity == [3]u32{0, 1, 0})
		assert(len(sel.classes) == 0)
		assert(~sel.has_something) // no callback
		assert(sel.previous == nil)

		props := rule.properties
		assert(len(props) == 2)
		assert(props[0].name == "color")
		assert(len(props[0].value) == 1)
		assert(props[1].name == "font-size")
		assert(len(props[1].value) == 1)
	}
}

@(test)
test_parser_2 :: proc(_: ^testing.T) {
	src := `
	b > a { color: red }
	b + a { color: green }
	a:lang(c) { color: blue }
	.class[attr=text] { color: #fff }
	`

	tzr := tokenize_string(src, {}, context.temp_allocator)
	style := parse(tzr.tokens[:], {}, context.temp_allocator)
	part: ^Selector_Part

	assert(len(style.rules) == 4)
	rule := style.rules[0]
	assert(len(rule.selectors) == 1)
	assert(rule.selectors[0].specificity == [3]u32{0, 0, 2})
	part = &rule.selectors[0].last
	assert(part.tag == "a")
	assert(len(part.classes) == 0)
	assert(part.has_something)
	assert(part.combinator == .Child)
	part = part.previous
	assert(part != nil)
	assert(part.tag == "b")
	assert(len(part.classes) == 0)
	assert(part.has_something)
	assert(part.previous == nil)

	rule = style.rules[1]
	assert(len(rule.selectors) == 1)
	assert(rule.selectors[0].specificity == [3]u32{0, 0, 2})
	part = &rule.selectors[0].last
	assert(part.tag == "a")
	assert(len(part.classes) == 0)
	assert(part.has_something)
	assert(part.combinator == .Next)
	part = part.previous
	assert(part != nil)
	assert(part.tag == "b")
	assert(len(part.classes) == 0)
	assert(part.has_something)
	assert(part.previous == nil)

	rule = style.rules[2]
	assert(len(rule.selectors) == 1)
	assert(rule.selectors[0].specificity == [3]u32{0, 1, 1})
	part = &rule.selectors[0].last
	assert(part.tag == "a")

	rule = style.rules[3]
	assert(len(rule.selectors) == 1)
	assert(rule.selectors[0].specificity == [3]u32{0, 2, 0})
	part = &rule.selectors[0].last
	assert(part.tag == "")
	assert(len(part.classes) == 1)
	assert(part.classes[0] == "class")
	assert(len(part.attributes) == 1)
	assert(part.attributes[0].name == "attr")
	assert(part.attributes[0].value == "text")
	assert(part.attributes[0].pattern == .Exact)
}

@(test)
test_parser_3 :: proc(_: ^testing.T) {
	src := `
    @font-face {
        font-family: "Stuff";
        src: url(there);
    }

    @media screen and (width >= 0)
    {* { --fg-color: #fff000; }
        should-not-exist {}
    }

    tag.class#id[attr*='text'], second * {
        color: #fff;
        background: linear(@fg-color, #000);
        transform: rotate(30deg);
        font-size: 120%;
    }

    #id::sub:not(:hover) >fourth .fifth {
        color: red;

        @media paper { color: white }
    }

    @define-drawable the-drawable 'the/path';
    `

	tzr := tokenize_string(src, {}, context.temp_allocator)
	style := parse(tzr.tokens[:], {}, context.temp_allocator)

	{
		assert(len(style.at_rules) == 3)
		{
			rule := style.at_rules[0]
			assert(rule.keyword == "font-face")
			assert(len(rule.prelude) == 0)
			assert(len(rule.contents) > 0)
		}
		{
			rule := style.at_rules[1]
			assert(rule.keyword == "media")
			assert(len(rule.prelude) > 0)
			assert(len(rule.contents) > 0)
		}
		{
			rule := style.at_rules[2]
			assert(rule.keyword == "define-drawable")
			assert(rule.prelude[0].text == "the-drawable")
			assert(rule.prelude[1].kind == .Whitespace)
			assert(rule.prelude[2].kind == .Str)
			assert(rule.prelude[2].text == "the/path")
		}
	}

	{
		part: ^Selector_Part

		assert(len(style.rules) == 2)
		rule := style.rules[0]
		assert(len(rule.selectors) == 2)
		assert(rule.selectors[0].specificity == [3]u32{1, 2, 1})
		part = &rule.selectors[0].last
		assert(part.tag == "tag")
		assert(part.id == "id")
		assert(len(part.classes) == 1)
		assert(part.classes[0] == "class")
		assert(len(part.attributes) == 1)
		assert(part.attributes[0].name == "attr")
		assert(part.attributes[0].value == "text")
		assert(part.attributes[0].pattern == .Substring)
		assert(len(part.functions) == 0)
		assert(part.has_something)
		part = &rule.selectors[1].last
		assert(rule.selectors[1].specificity == [3]u32{0, 0, 1})
		assert(!part.has_something)
		assert(part.combinator == .Descendant)
		assert(part.previous != nil)
		part = part.previous
		assert(part.tag == "second")
		assert(part.has_something)
		assert(part.previous == nil)

		props := rule.properties
		assert(len(props) == 4)
		assert(props[0].name == "color")
		assert(len(props[0].value) == 1)
		assert(props[1].name == "background")
		assert(len(props[1].value) == 5)
		assert(props[2].name == "transform")
		assert(len(props[2].value) == 3)
		assert(props[3].name == "font-size")
		assert(len(props[3].value) == 1)

		rule = style.rules[1]
		assert(len(rule.selectors) == 1)
		assert(rule.selectors[0].specificity == [3]u32{1, 2, 2})
		part = &rule.selectors[0].last
		assert(part.tag == "")
		assert(part.id == "")
		assert(len(part.classes) == 1)
		assert(part.classes[0] == "fifth")
		assert(len(part.attributes) == 0)
		assert(part.has_something)
		assert(part.combinator == .Descendant)
		part = part.previous
		assert(part != nil)
		assert(part.tag == "fourth")
		assert(len(part.classes) == 0)
		assert(len(part.attributes) == 0)
		part.combinator = .Child
		part = part.previous
		assert(part != nil)
		assert(part.tag == "")
		assert(part.id == "id")
		assert(part.pseudo_element == "sub")
		assert(len(part.functions) == 0)
		assert(part.not_function != nil)
		assert(part.previous == nil)

		assert(len(rule.properties) == 1)
		assert(rule.properties[0].name == "color")
		assert(len(rule.nested_at_rules) == 1)
		assert(rule.nested_at_rules[0].keyword == "media")
	}
}

@(test)
test_parser_4 :: proc(_: ^testing.T) {
	src := `
	:is(.a, a.b) c {
		color: red;
	}
	head :where(.a, a.b) c {
		color: red;
	}
	a:not([href]) {
		color: red;
	}
	a:not(:hover, :active, ::invalid) {
		color: red;
	}
	:nth-child(2n+1 of :not([hidden])) {
		color: red;
	}
	:host(#id) {
		color: red;
	}
	`

	tzr := tokenize_string(src, {}, context.temp_allocator)
	style := parse(tzr.tokens[:], {}, context.temp_allocator)
	part: ^Selector_Part

	rule := style.rules[0]
	assert(len(rule.selectors) == 1)
	assert(rule.selectors[0].specificity == [3]u32{0, 1, 2})
	part = &rule.selectors[0].last
	assert(part.tag == "c")
	assert(len(part.functions) == 0)
	assert(part.previous != nil)
	part = part.previous
	assert(part.tag == "")
	assert(len(part.functions) == 1)
	assert(part.functions[0].func == .Is)
	assert(part.functions[0].name == "is")
	assert(len(part.functions[0].selectors) == 2)
	assert(part.functions[0].selectors[0].tag == "")
	assert(part.functions[0].selectors[1].tag == "a")
	assert(len(part.functions[0].selectors[0].classes) == 1)
	assert(len(part.functions[0].selectors[1].classes) == 1)
	assert(part.functions[0].selectors[0].classes[0] == "a")
	assert(part.functions[0].selectors[1].classes[0] == "b")
	assert(part.previous == nil)

	rule = style.rules[1]
	assert(len(rule.selectors) == 1)
	assert(rule.selectors[0].specificity == [3]u32{0, 0, 2})
	part = &rule.selectors[0].last
	assert(part.tag == "c")
	assert(len(part.functions) == 0)
	assert(part.previous != nil)
	part = part.previous
	assert(part.tag == "")
	assert(len(part.functions) == 1)
	assert(part.functions[0].func == .Is)
	assert(part.functions[0].name == "where")
	assert(len(part.functions[0].selectors) == 2)
	assert(part.functions[0].selectors[0].tag == "")
	assert(part.functions[0].selectors[1].tag == "a")
	assert(len(part.functions[0].selectors[0].classes) == 1)
	assert(len(part.functions[0].selectors[1].classes) == 1)
	assert(part.functions[0].selectors[0].classes[0] == "a")
	assert(part.functions[0].selectors[1].classes[0] == "b")
	part = part.previous
	assert(part.tag == "head")
	assert(len(part.functions) == 0)
	assert(part.previous == nil)

	rule = style.rules[2]
	assert(len(rule.selectors) == 1)
	assert(rule.selectors[0].specificity == [3]u32{0, 1, 1})
	part = &rule.selectors[0].last
	assert(part.tag == "a")
	assert(len(part.functions) == 0)
	assert(part.not_function != nil)
	assert(len(part.not_function.attributes) == 1)
	assert(part.not_function.attributes[0].name == "href")
	assert(part.not_function.attributes[0].value == "")
	assert(part.previous == nil)

	rule = style.rules[3]
	assert(len(rule.selectors) == 1)
	assert(rule.selectors[0].specificity == [3]u32{0, 1, 1})
	part = &rule.selectors[0].last
	assert(part.tag == "a")
	assert(len(part.functions) == 1)
	assert(part.functions[0].func == .Not)
	assert(part.functions[0].name == "not")
	assert(len(part.functions[0].selectors) == 2)
	assert(part.functions[0].selectors[0].tag == "")
	assert(part.functions[0].selectors[1].tag == "")
	assert(part.previous == nil)

	rule = style.rules[4]
	assert(len(rule.selectors) == 1)
	assert(rule.selectors[0].specificity == [3]u32{0, 2, 0})
	part = &rule.selectors[0].last
	assert(len(part.functions) == 1)
	assert(part.functions[0].func == .Nth)
	assert(part.functions[0].name == "nth-child")
	assert(len(part.functions[0].selectors) == 1)
	assert(len(part.functions[0].selectors[0].functions) == 0)
	assert(part.functions[0].selectors[0].not_function != nil)
	assert(len(part.functions[0].selectors[0].not_function.attributes) == 1)
	assert(part.functions[0].selectors[0].not_function.attributes[0].name == "hidden")
	assert(part.previous == nil)

	rule = style.rules[5]
	assert(len(rule.selectors) == 1)
	assert(rule.selectors[0].specificity == [3]u32{1, 1, 0})
	part = &rule.selectors[0].last
	assert(len(part.functions) == 1)
	assert(part.functions[0].func == .Host)
	assert(part.functions[0].name == "host")
	assert(len(part.functions[0].selectors) == 1)
	assert(part.functions[0].selectors[0].id == "id")
	assert(part.previous == nil)
}

@(test)
test_parser_nesting :: proc(_: ^testing.T) {
	src := `
		.test {
			& { padding: 1ch }
			&& { padding: 2ch }
			* { font-size: 1rem }
			.bar { font-size: 1.5rem }
			.bar.bar { font-size: 1.5rem }
			#id { font-size: 2rem }
			[href] { color: blue }
			&:hover { color: black }
			& :hover { color: red }

			& button { padding: 1em }
			& button:hover { padding: 2em }

			.second {
				.third { padding-left: 10px }
			}
		}

		p, div {
			padding: 1ch;

			> span, + span, ~ span, & span {
				color: orange;
			}

			margin: 2ch;

			.component &,
			.outermost & .innermost {
				padding-left: 0;
			}
		}

		#a, b {
			& c { color: blue; }
		}
		.test c { color: red; }
	`

	tzr := tokenize_string(src, {}, context.temp_allocator)
	style := parse(tzr.tokens[:], {}, context.temp_allocator)

	// fmt.printfln("%#w", style.rules[0])
}

@(test)
test_specificity :: proc(_: ^testing.T) {
	assert(!less_specific({}, {}))
	assert(less_specific({0, 0, 1}, {0, 0, 2}))
	assert(less_specific({0, 1, 0}, {0, 2, 0}))
	assert(less_specific({1, 0, 0}, {2, 0, 0}))
	assert(less_specific({0, 0, 1}, {0, 1, 0}))
	assert(less_specific({0, 1, 0}, {1, 0, 0}))
	assert(less_specific({1, 1, 0}, {2, 0, 0}))
	assert(!less_specific({1, 1, 1}, {1, 0, 0}))
}
