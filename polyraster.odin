// SPDX-FileCopyrightText: © 2025 Viktor M. <dayllenger@gmail.com>
// SPDX-License-Identifier: Zlib
//
// A small software rasterizer for 2D vector graphics. Works with complex
// polygons, trapezoids, and thin 1px lines.
//
// Polygon rasterizer is built upon stb_truetype.h rasterizers by Sean Barrett.
// Clipping and line rasterization are mostly standard algorithms.
package polyraster

import "core:fmt"
import "core:math"

Vec2 :: [2]f32

BoxI :: struct {
	x, y, w, h: int,
}

FillRule :: enum {
	Nonzero,
	Odd,
	Zero,
	Even,
}

RastParams :: struct {
	antialias: bool,
	clip:      BoxI,
	rule:      FillRule,
}

Plotter :: struct {
	set_pixel:     proc(self: ^Plotter, x, y: int),
	mix_pixel:     proc(self: ^Plotter, x, y: int, alpha: f32),
	set_scan_line: proc(self: ^Plotter, x1, x2, y: int),
	mix_scan_line: proc(self: ^Plotter, x1, x2, y: int, alpha: f32),
}

@(private)
RectI :: struct {
	left, top, right, bottom: int,
}

@(private)
Rect :: struct {
	left, top, right, bottom: f32,
}

@(private)
boxi_from_rect :: proc(r: RectI) -> BoxI {
	return {r.left, r.top, r.right - r.left, r.bottom - r.top}
}

@(private)
recti_from_box :: proc(b: BoxI) -> RectI {
	return {b.x, b.y, b.x + b.w, b.y + b.h}
}

@(private)
recti_intersect :: proc(a: ^RectI, b: RectI) {
	if a.left < b.left {a.left = b.left}
	if a.top < b.top {a.top = b.top}
	if a.right > b.right {a.right = b.right}
	if a.bottom > b.bottom {a.bottom = b.bottom}
}

@(private)
plotter_has_all_methods :: proc(plotter: ^Plotter) -> bool {
	return(
		plotter != nil &&
		plotter.set_pixel != nil &&
		plotter.mix_pixel != nil &&
		plotter.set_scan_line != nil &&
		plotter.mix_scan_line != nil \
	)
}

@(private)
compute_bounding_box :: proc(points: []Vec2) -> Rect {
	r := Rect{max(f32), max(f32), min(f32), min(f32)}
	for p in points {
		if r.left > p.x {r.left = p.x}
		if r.top > p.y {r.top = p.y}
		if r.right < p.x {r.right = p.x}
		if r.bottom < p.y {r.bottom = p.y}
	}
	return r
}

@(private)
is_fill_rule_inverted :: proc(rule: FillRule) -> bool {
	return rule == .Zero || rule == .Even
}

@(private)
EPS :: 0.01

// --- Polygon rasterizer ---

rasterize_polygons :: proc(points: []Vec2, contours: []i32, params: RastParams, plotter: ^Plotter) {
	if len(points) < 3 && !is_fill_rule_inverted(params.rule) {
		return
	}

	contours := contours
	if len(contours) == 0 {
		contours = []i32{i32(len(points))}
	}

	assert(params.clip.w > 0 && params.clip.h > 0)
	assert(plotter_has_all_methods(plotter))
	params := params

	// perform clipping first; contours may change their geometry

	Contour :: struct {
		start:   int,
		length:  int,
		clipped: bool,
		bounds:  RectI,
	}
	ctrs := make([]Contour, len(contours), context.temp_allocator)
	if ctrs == nil {return}

	// we will adjust rasterizer clipping box to the polygon bounding box,
	// because clip is also the working area (on usual fill rules),
	// and we need to minimize it
	clip := recti_from_box(params.clip)
	bbox := RectI{max(int), max(int), min(int), min(int)}
	pts := make([dynamic]Vec2, context.temp_allocator)

	// initialize the array, collect contour bounding boxes into it
	n: int
	for length, i in contours {
		ctr: ^Contour = &ctrs[i]
		ctr.start = n
		ctr.length = int(length)
		ctr.clipped = false
		cb: Rect = compute_bounding_box(points[n:][:length])
		cbi: RectI = {ifloor(cb.left), ifloor(cb.top), iceil(cb.right), iceil(cb.bottom)}
		ctr.bounds = cbi
		bbox.left = min(bbox.left, cbi.left)
		bbox.top = min(bbox.top, cbi.top)
		bbox.right = max(bbox.right, cbi.right)
		bbox.bottom = max(bbox.bottom, cbi.bottom)
		n += int(length)
	}

	// clip contours
	for &ctr, i in ctrs {
		cb: RectI = ctr.bounds
		if cb.top >= clip.bottom || cb.left > clip.right || cb.bottom <= clip.top || cb.right <= clip.left {
			ctr.start = 0
			ctr.length = 0
			ctr.clipped = true
			continue
		}

		clipf := Rect{f32(clip.left), f32(clip.top), f32(clip.right), f32(clip.bottom)}
		clip_edges: struct {
			data:  [4][2]Vec2,
			count: int,
		}
		if cb.top < clip.top {
			clip_edges.data[clip_edges.count] = {Vec2{clipf.left, clipf.top}, Vec2{clipf.right, clipf.top}}
			clip_edges.count += 1
		}
		if cb.right > clip.right {
			clip_edges.data[clip_edges.count] = {Vec2{clipf.right, clipf.top}, Vec2{clipf.right, clipf.bottom}}
			clip_edges.count += 1
		}
		if cb.bottom > clip.bottom {
			clip_edges.data[clip_edges.count] = {Vec2{clipf.right, clipf.bottom}, Vec2{clipf.left, clipf.bottom}}
			clip_edges.count += 1
		}
		if cb.left < clip.left {
			clip_edges.data[clip_edges.count] = {Vec2{clipf.left, clipf.bottom}, Vec2{clipf.left, clipf.top}}
			clip_edges.count += 1
		}
		if clip_edges.count > 0 {
			before := len(pts)
			clip_polygon(points[ctr.start:][:ctr.length], clip_edges.data[:clip_edges.count], &pts)
			ctr.start = before
			ctr.length = len(pts) - before
			ctr.clipped = true
		}
	}
	if !is_fill_rule_inverted(params.rule) {
		recti_intersect(&bbox, clip)
		params.clip = boxi_from_rect(bbox)
	}

	// now we have to blow out the windings into explicit edge lists
	n = 0
	for ctr in ctrs {
		n += ctr.length
	}
	if n == 0 {return}

	edges := make([]Edge, n + 1, context.temp_allocator) // add an extra one as a sentinel
	if edges == nil {return}
	n = 0

	for &ctr in ctrs {
		if ctr.length == 0 {continue}

		p: [^]Vec2 = ctr.clipped ? &pts[ctr.start] : &points[ctr.start]
		j := ctr.length - 1

		for k := 0; k < ctr.length; k += 1 {
			// skip the edge if horizontal
			if p[j].y == p[k].y {
				j = k
				continue
			}

			// add edge from j to k to the list
			a, b := k, j
			e: Edge
			if p[j].y < p[k].y {
				e.invert = true
				a, b = j, k
			}
			e.x0 = p[a].x - f32(params.clip.x)
			e.y0 = p[a].y - f32(params.clip.y)
			e.x1 = p[b].x - f32(params.clip.x)
			e.y1 = p[b].y - f32(params.clip.y)
			edges[n] = e
			n += 1
			j = k
		}
	}

	// now sort the edges by their highest point (should snap to integer, and then by x)
	sort_edges(edges, n)

	// now, traverse the scanlines and find the intersections on each scanline
	if params.antialias {
		rasterize_sorted_edges_aa(edges, n, params, plotter)
	} else {
		rasterize_sorted_edges_no_aa(edges, n, params, plotter)
	}
}

// Perform Sutherland-Hodgman clipzping of the input 2D polygon inside a set of edges.
//
// The rasterizer only clips by an axis-aligned rectangle but this algorithm
// supports any convex clip polygon.
clip_polygon :: proc(input: []Vec2, clip_edges: [][2]Vec2, output: ^[dynamic]Vec2) {
	// check if a point is on right side of an edge
	is_inside :: proc(p: Vec2, edge: [2]Vec2) -> bool {
		a, b := edge[0], edge[1]
		return cross2d(b - a, p - a) > 0
	}

	// calculate intersection point
	intersect :: proc(edge: [2]Vec2, s, e: Vec2) -> Vec2 {
		a, b := edge[0], edge[1]
		es := s - e
		ba := a - b
		return (es * cross2d(a, b) - ba * cross2d(s, e)) * (1 / cross2d(ba, es))
	}

	cross2d :: proc(a, b: Vec2) -> f32 {
		return a.x * b.y - a.y * b.x
	}

	// double bufferization to avoid extra copying for each clip edge
	tmp := make([dynamic]Vec2, context.temp_allocator)
	input := input
	initial_len := len(output)
	buf: ^[dynamic]Vec2 = (len(clip_edges) % 2 == 1) ? output : &tmp

	for &edge in clip_edges {
		// iterate subject polygon edges
		for s, i in input {
			e: Vec2 = input[(i + 1) % len(input)]

			if is_inside(s, edge) && is_inside(e, edge) {
				// Case 1: Both vertices are inside:
				// Only the second vertex is added to the output list
				append(buf, e)
			} else if !is_inside(s, edge) && is_inside(e, edge) {
				// Case 2: First vertex is outside while second one is inside:
				// Both the point of intersection of the edge with the clip boundary
				// and the second vertex are added to the output list
				append(buf, intersect(edge, s, e))
				append(buf, e)
			} else if is_inside(s, edge) && !is_inside(e, edge) {
				// Case 3: First vertex is inside while second one is outside:
				// Only the point of intersection of the edge with the clip boundary
				// is added to the output list
				append(buf, intersect(edge, s, e))
			} else {
				// Case 4: Both vertices are outside
				// No vertices are added to the output list
			}
		}

		// swap and reset the buffers
		if buf == output {
			input = output[initial_len:]
			buf = &tmp
			resize(&tmp, 0)
		} else {
			input = tmp[:]
			buf = output
			resize(output, initial_len)
		}
	}
}

@(private)
Edge :: struct {
	x0, y0: f32,
	x1, y1: f32,
	invert: bool,
}

@(private)
sort_edges :: proc(p: []Edge, n: int) {
	sort_edges_quicksort(p, n)
	sort_edges_ins_sort(p, n)
}

@(private)
cmp_edges :: proc(a, b: Edge) -> bool {
	return a.y0 < b.y0
}

@(private)
sort_edges_ins_sort :: proc(p: []Edge, n: int) {
	for i in 1 ..< n {
		t: Edge = p[i]
		j: int
		for j = i; j > 0; j -= 1 {
			if !cmp_edges(t, p[j - 1]) {
				break
			}
			p[j] = p[j - 1]
		}
		if i != j {
			p[j] = t
		}
	}
}

@(private)
sort_edges_quicksort :: proc(p: []Edge, n: int) {
	p, n := p, n
	// threshold for transitioning to insertion sort
	for n > 12 {
		// compute median of three
		m := n >> 1
		c01: bool = cmp_edges(p[0], p[m])
		c12: bool = cmp_edges(p[m], p[n - 1])
		// if 0 >= mid >= end, or 0 < mid < end, then use mid
		if c01 != c12 {
			// otherwise, we'll need to swap something else to middle
			c: bool = cmp_edges(p[0], p[n - 1])
			// 0>mid && mid<n:  0>n => n; 0<n => 0
			// 0<mid && mid>n:  0>n => 0; 0<n => n
			z := (c == c12) ? 0 : n - 1
			p[z], p[m] = p[m], p[z]
		}
		// now p[m] is the median-of-three
		// swap it to the beginning so it won't move around
		p[0], p[m] = p[m], p[0]

		// partition loop
		i := 1
		j := n - 1
		for {
			// handling of equality is crucial here
			// for sentinels & efficiency with duplicates
			for ;; i += 1 {
				if !cmp_edges(p[i], p[0]) {break}
			}
			for ;; j -= 1 {
				if !cmp_edges(p[0], p[j]) {break}
			}
			// make sure we haven't crossed
			if i >= j {
				break
			}
			p[i], p[j] = p[j], p[i]

			i += 1
			j -= 1
		}

		// recurse on smaller side, iterate on larger
		if j < (n - i) {
			sort_edges_quicksort(p, j)
			p = p[i:]
			n = n - i
		} else {
			sort_edges_quicksort(p[i:], n - i)
			n = j
		}
	}
}

@(private)
Hheap_chunk :: struct {
	next: ^Hheap_chunk,
}

@(private)
Hheap :: struct {
	head:                        ^Hheap_chunk,
	first_free:                  rawptr,
	num_remaining_in_head_chunk: int,
}

@(private)
hheap_alloc :: proc(hh: ^Hheap, size: int) -> rawptr {
	if hh.first_free != nil {
		ptr := hh.first_free
		hh.first_free = (cast(^rawptr)ptr)^
		return ptr
	}
	if hh.num_remaining_in_head_chunk == 0 {
		count := size < 32 ? 2000 : size < 128 ? 800 : 100
		buf := make([]byte, size_of(Hheap_chunk) + size * count)
		if buf == nil {
			return nil
		}
		c := cast(^Hheap_chunk)raw_data(buf)
		c.next = hh.head
		hh.head = c
		hh.num_remaining_in_head_chunk = count
	}
	hh.num_remaining_in_head_chunk -= 1
	return (cast([^]byte)hh.head)[size_of(Hheap_chunk) + size * hh.num_remaining_in_head_chunk:]
}

@(private)
hheap_free :: proc(hh: ^Hheap, ptr: rawptr) {
	(cast(^rawptr)ptr)^ = hh.first_free
	hh.first_free = ptr
}

@(private)
hheap_cleanup :: proc(hh: ^Hheap) {
	c: ^Hheap_chunk = hh.head
	for c != nil {
		n: ^Hheap_chunk = c.next
		free(c)
		c = n
	}
}

// antialiased part

@(private)
ActiveEdgeAA :: struct {
	next:         ^ActiveEdgeAA,
	fx, fdx, fdy: f32,
	direction:    f32,
	sy:           f32,
	ey:           f32,
}

@(private)
new_active_aa :: proc(hh: ^Hheap, e: ^Edge, start_point: f32) -> ^ActiveEdgeAA {
	z := cast(^ActiveEdgeAA)hheap_alloc(hh, size_of(ActiveEdgeAA))
	dxdy: f32 = (e.x1 - e.x0) / (e.y1 - e.y0)
	assert(z != nil); if z == nil {return nil}

	z.fdx = dxdy
	z.fdy = dxdy != 0.0 ? (1.0 / dxdy) : 0.0
	z.fx = e.x0 + dxdy * (start_point - e.y0)
	z.direction = e.invert ? 1.0 : -1.0
	z.sy = e.y0
	z.ey = max(e.y1, start_point) // avoid a floating-point precision problem
	z.next = nil
	return z
}

// directly AA rasterize edges w/o supersampling
@(private)
rasterize_sorted_edges_aa :: proc(edges: []Edge, n: int, params: RastParams, plotter: ^Plotter) {
	hh: Hheap
	active: ^ActiveEdgeAA
	defer hheap_cleanup(&hh)

	scanline_data: [512 + 1]f32 = ---
	scanline, scanline2: [^]f32
	width: int = params.clip.w

	if width > 256 {
		scanline = raw_data(make([]f32, width * 2 + 1))
	} else {
		scanline = raw_data(scanline_data[:])
	}
	scanline2 = scanline[width:]
	defer if scanline != raw_data(scanline_data[:]) {free(scanline)}

	edges := edges
	edges[n].y0 = f32(params.clip.h + 1)

	for y in 0 ..< params.clip.h {
		// find scanline Y bounds
		scan_y_top := f32(y)
		scan_y_bottom := f32(y) + 1.0
		step: ^^ActiveEdgeAA = &active

		// update all active edges;
		// remove all active edges that terminate before the top of this scanline
		for step^ != nil {
			z: ^ActiveEdgeAA = step^
			if z.ey <= scan_y_top {
				step^ = z.next // delete from list
				assert(z.direction != 0.0)
				z.direction = 0
				hheap_free(&hh, z)
			} else {
				step = &((step^).next) // advance through list
			}
		}

		// insert all edges that start before the bottom of this scanline
		for edges[0].y0 <= scan_y_bottom {
			if edges[0].y0 != edges[0].y1 {
				z: ^ActiveEdgeAA = new_active_aa(&hh, &edges[0], scan_y_top)
				if z != nil {
					// insert at front
					z.next = active
					active = z
				}
			}
			edges = edges[1:]
		}

		// now process all active edges
		if active != nil {
			for x in 0 ..< width {scanline[x] = 0}
			for x in 0 ..< width + 1 {scanline2[x] = 0}

			span: [2]int = fill_active_edges_aa(scanline, scanline2[1:], width, active, scan_y_top)

			xx: int = params.clip.x
			yy: int = params.clip.y + y
			// the calls to the next function should get inlined
			switch params.rule {
			case .Nonzero:
				cov := proc(w: f32) -> f32 {return abs(w)}
				draw_scanline_aa(scanline, cov, width, xx, yy, span, plotter)
			case .Odd:
				cov := proc(w: f32) -> f32 {return w == 0 ? 0 : abs(w - math.round(w / 2) * 2)}
				draw_scanline_aa(scanline, cov, width, xx, yy, span, plotter)
			case .Zero:
				cov := proc(w: f32) -> f32 {return w == 0 ? 1 : (1 - min(abs(w), 1))}
				draw_scanline_aa(scanline, cov, width, xx, yy, {0, width}, plotter)
			case .Even:
				cov := proc(w: f32) -> f32 {return w == 0 ? 1 : (1 - abs(w - math.round(w / 2) * 2))}
				draw_scanline_aa(scanline, cov, width, xx, yy, {0, width}, plotter)
			}
		} else if is_fill_rule_inverted(params.rule) {
			// fill outer areas
			plotter->set_scan_line(params.clip.x, params.clip.x + width, params.clip.y + y)
		}

		// advance all the edges
		step = &active
		for step^ != nil {
			z: ^ActiveEdgeAA = step^
			z.fx += z.fdx // advance to position for current scanline
			step = &((step^).next) // advance through list
		}
	}
}

// returns filled scanline boundaries
@(private)
fill_active_edges_aa :: proc(scanline, scanline_fill: [^]f32, len: int, e: ^ActiveEdgeAA, y_top: f32) -> [2]int {
	e := e
	y_bottom: f32 = y_top + 1
	fx0, fx1: f32 = f32(len), 0

	for e != nil {
		// brute force every pixel

		// compute intersection points with top & bottom
		assert(e.ey >= y_top)

		if e.fdx != 0 {
			x0: f32 = e.fx
			dx: f32 = e.fdx
			xb: f32 = x0 + dx
			x_top, x_bottom: f32
			sy0, sy1: f32
			dy: f32 = e.fdy
			assert(e.sy <= y_bottom && e.ey >= y_top)

			// compute endpoints of line segment clipped to this scanline (if the
			// line segment starts on this scanline). x0 is the intersection of the
			// line with y_top, but that may be off the line segment.
			if e.sy > y_top {
				x_top = x0 + dx * (e.sy - y_top)
				sy0 = e.sy
			} else {
				x_top = x0
				sy0 = y_top
			}
			if e.ey < y_bottom {
				x_bottom = x0 + dx * (e.ey - y_top)
				sy1 = e.ey
			} else {
				x_bottom = xb
				sy1 = y_bottom
			}
			// after clipping, the endpoint can appear out of bounds a bit
			if x_top < 0 {x_top = 0}
			if x_top >= f32(len) {x_top = f32(len) - EPS}
			if x_bottom < 0 {x_bottom = 0}
			if x_bottom >= f32(len) {x_bottom = f32(len) - EPS}

			// from here on, we don't have to range check x values

			if int(x_top) == int(x_bottom) {
				// simple case, only spans one pixel
				x := int(x_top)
				height: f32 = sy1 - sy0
				scanline[x] += e.direction * (1 - ((x_top - f32(x)) + (x_bottom - f32(x))) / 2) * height
				scanline_fill[x] += e.direction * height // everything right of this pixel is filled
			} else {
				// covers 2+ pixels
				if x_top > x_bottom {
					// flip scanline vertically; signed area is the same
					sy0 = y_bottom - (sy0 - y_top)
					sy1 = y_bottom - (sy1 - y_top)
					sy0, sy1 = sy1, sy0
					x_top, x_bottom = x_bottom, x_top
					dy = -dy
					x0, xb = xb, x0
				}

				x1 := int(x_top)
				x2 := int(x_bottom)
				// compute intersection with y axis at x1+1
				y_crossing: f32 = (f32(x1) + 1 - x0) * dy + y_top

				sign: f32 = e.direction
				// area of the rectangle covered from y0..y_crossing
				area: f32 = sign * (y_crossing - sy0)
				// area of the triangle (x_top,y0), (x+1,y0), (x+1,y_crossing)
				scanline[x1] += area * (1 - ((x_top - f32(x1)) + f32(x1 + 1 - x1)) / 2)

				step: f32 = sign * dy
				for x in x1 + 1 ..< x2 {
					scanline[x] += area + step / 2
					area += step
				}
				y_crossing += dy * f32(x2 - (x1 + 1))

				area = min(area, 1)

				scanline[x2] += area + sign * (1 - (f32(x2 - x2) + (x_bottom - f32(x2))) / 2) * (sy1 - y_crossing)
				scanline_fill[x2] += sign * (sy1 - sy0)
			}
		} else {
			// fully vertical
			// simplified version of the code above
			x0: f32 = e.fx
			if x0 < 0 {x0 = 0}
			if x0 >= f32(len) {x0 = f32(len) - EPS}

			x := int(x0)
			height: f32 = min(e.ey, y_bottom) - max(e.sy, y_top)
			scanline[x] += e.direction * (1 - (x0 - f32(x))) * height
			scanline_fill[x] += e.direction * height
		}

		// find line boundaries for optimization
		fx0 = min(fx0, e.fx, e.fx + e.fdx)
		fx1 = max(fx1, e.fx, e.fx + e.fdx)

		e = e.next
	}
	return {max(ifloor(fx0), 0), min(iceil(fx1), len)}
}

@(private)
draw_scanline_aa :: #force_inline proc(
	scanline: [^]f32,
	calc_coverage: proc(w: f32) -> f32,
	width, x, y: int,
	span: [2]int,
	plotter: ^Plotter,
) {
	scanline2: [^]f32 = scanline[width:]
	prev: int = x
	run: bool
	sum: f32

	for i in span[0] ..< span[1] {
		sum += scanline2[i]
		cov: f32 = calc_coverage(scanline[i] + sum)
		if cov > 1 - EPS {
			if !run {
				prev = x + i
				run = true
			}
		} else {
			xx: int = x + i
			if run {
				plotter->set_scan_line(prev, xx, y)
				run = false
			}
			if cov > EPS {
				plotter->mix_pixel(xx, y, cov)
			}
		}
	}
	if run {
		plotter->set_scan_line(prev, x + span[1], y)
	}
}

// non-antialiased part

@(private)
FIXSHIFT :: 10
@(private)
FIX :: 1 << FIXSHIFT
@(private)
FIXHALF :: 1 << (FIXSHIFT - 1)
@(private)
FIXMASK :: FIX - 1

@(private)
ActiveEdge :: struct {
	next:      ^ActiveEdge,
	x, dx:     int,
	ey:        f32,
	direction: int,
}

@(private)
new_active :: proc(hh: ^Hheap, e: ^Edge, start_point: f32) -> ^ActiveEdge {
	z := cast(^ActiveEdge)hheap_alloc(hh, size_of(ActiveEdge))
	dxdy: f32 = (e.x1 - e.x0) / (e.y1 - e.y0)
	assert(z != nil); if z == nil {return nil}

	// round dx down to avoid overshooting
	if dxdy < 0 {
		z.dx = -ifloor(FIX * -dxdy)
	} else {
		z.dx = ifloor(FIX * dxdy)
	}

	// use z.dx so when we offset later it's by the same amount
	z.x = ifloor(FIX * e.x0 + f32(z.dx) * (start_point - e.y0))

	z.ey = e.y1
	z.next = nil
	z.direction = e.invert ? 1 : -1
	return z
}

@(private)
sort_active_edges_bubble :: proc(active: ^^ActiveEdge) {
	step: ^^ActiveEdge
	for {
		changed: bool
		step = active

		for step^ != nil && (step^).next != nil {
			if (step^).x > (step^).next.x {
				t: ^ActiveEdge = step^
				q: ^ActiveEdge = t.next

				t.next = q.next
				q.next = t
				step^ = q
				changed = true
			}
			step = &(step^).next
		}
		if !changed {
			break
		}
	}
}

@(private)
sort_active_edges_merge :: proc(head: ^ActiveEdge) -> ^ActiveEdge {
	if head == nil || head.next == nil {
		return head
	}

	front_back_split :: proc(source: ^ActiveEdge) -> (front, back: ^ActiveEdge) {
		slow: ^ActiveEdge = source
		fast: ^ActiveEdge = source.next

		for fast != nil {
			fast = fast.next
			if fast != nil {
				slow = slow.next
				fast = fast.next
			}
		}

		front, back = source, slow.next
		slow.next = nil
		return front, back
	}

	sorted_merge :: proc(a, b: ^ActiveEdge) -> ^ActiveEdge {
		if a == nil {return b}
		if b == nil {return a}

		result: ^ActiveEdge
		if a.x < b.x {
			result = a
			result.next = sorted_merge(a.next, b)
		} else {
			result = b
			result.next = sorted_merge(a, b.next)
		}
		return result
	}

	a, b: ^ActiveEdge = front_back_split(head)

	a = sort_active_edges_merge(a)
	b = sort_active_edges_merge(b)

	return sorted_merge(a, b)
}

@(private)
rasterize_sorted_edges_no_aa :: proc(edges: []Edge, n: int, params: RastParams, plotter: ^Plotter) {
	hh: Hheap
	active: ^ActiveEdge
	defer hheap_cleanup(&hh)

	edges := edges
	edges[n].y0 = f32(params.clip.h + 1)

	for y in 0 ..< params.clip.h {
		// find center of pixel for this scanline
		scan_y := f32(y) + 0.5
		step: ^^ActiveEdge = &active

		// update all active edges;
		// remove all active edges that terminate before the center of this scanline
		count: int
		for step^ != nil {
			z: ^ActiveEdge = step^
			if z.ey <= scan_y {
				step^ = z.next // delete from list
				assert(z.direction != 0)
				z.direction = 0
				hheap_free(&hh, z)
			} else {
				z.x += z.dx // advance to position for current scanline
				step = &z.next // advance through list
			}
			count += 1
		}

		// re-sort the list if needed
		if count < 20 {
			sort_active_edges_bubble(&active)
		} else {
			active = sort_active_edges_merge(active)
		}

		// insert all edges that start before the center of this scanline -- omit ones that also end on this scanline
		for edges[0].y0 <= scan_y {
			if edges[0].y1 > scan_y {
				z: ^ActiveEdge = new_active(&hh, &edges[0], scan_y)
				if z != nil {
					// find insertion point
					if active == nil {
						active = z
					} else if z.x < active.x {
						// insert at front
						z.next = active
						active = z
					} else {
						// find thing to insert AFTER
						p: ^ActiveEdge = active
						for p.next != nil && p.next.x < z.x {
							p = p.next
						}
						// at this point, p.next.x is NOT < z.x
						z.next = p.next
						p.next = z
					}
				}
			}
			edges = edges[1:]
		}

		// now process all active edges
		if active != nil {
			len: int = params.clip.w
			xx: int = params.clip.x
			yy: int = params.clip.y + y
			// the calls to the next function should get inlined
			switch params.rule {
			case .Nonzero:
				fill_active_edges_no_aa(len, active, proc(w: int) -> bool {return w != 0}, xx, yy, plotter)
			case .Odd:
				fill_active_edges_no_aa(len, active, proc(w: int) -> bool {return w % 2 != 0}, xx, yy, plotter)
			case .Zero:
				fill_active_edges_no_aa(len, active, proc(w: int) -> bool {return w == 0}, xx, yy, plotter)
			case .Even:
				fill_active_edges_no_aa(len, active, proc(w: int) -> bool {return w % 2 == 0}, xx, yy, plotter)
			}
		} else if is_fill_rule_inverted(params.rule) {
			// fill outer areas
			plotter->set_scan_line(params.clip.x, params.clip.x + params.clip.w, params.clip.y + y)
		}
	}
}

@(private)
fill_active_edges_no_aa :: #force_inline proc(
	len: int,
	e: ^ActiveEdge,
	is_filled: proc(w: int) -> bool,
	x, y: int,
	plotter: ^Plotter,
) {
	e := e
	x0, w: int

	for e != nil {
		if !is_filled(w) {
			// if we're currently in the unfilled area, we need to record the edge start point
			x0 = e.x
			w += e.direction
		} else {
			x1: int = e.x
			w += e.direction
			// if we went to the unfilled area, we need to draw
			if !is_filled(w) {
				draw_scanline_no_aa(len, x0, x1, x, y, plotter)
			}
		}
		e = e.next
	}

	// fill the rest (only in inverted fill rules)
	if is_filled(w) {
		draw_scanline_no_aa(len, x0, len << FIXSHIFT, x, y, plotter)
	}
}

@(private)
draw_scanline_no_aa :: proc(len: int, x0, x1, x, y: int, plotter: ^Plotter) {
	i: int = x0 >> FIXSHIFT
	j: int = x1 >> FIXSHIFT

	if i < len && j >= 0 {
		if i != j {
			if i >= 0 {
				// check x0 coverage
				if FIX - (x0 & FIXMASK) < FIXHALF {
					i += 1
				}
			} else {
				i = 0 // clip
			}

			if j < len {
				// check x1 coverage
				if (x1 & FIXMASK) >= FIXHALF {
					j += 1
				}
			} else {
				j = len // clip
			}

			// fill pixels between x0 and x1
			plotter->set_scan_line(x + i, x + j, y)
		} else {
			// x0,x1 are the same pixel, so compute combined coverage
			if x1 - x0 >= FIXHALF {
				xx := x + i
				plotter->set_scan_line(xx, xx + 1, y)
			}
		}
	}
}

// --- Trapezoid rasterizer ---

HorizEdge :: struct {
	l, r, y: f32,
}

is_valid_trapezoid :: proc(top, bot: HorizEdge) -> bool {
	if top.y > bot.y {
		return false // a trapezoid with zero height is a chain break
	}
	return (top.l + EPS < top.r && bot.l <= bot.r) || (bot.l + EPS < bot.r && top.l <= top.r)
}

// Convert a simple y-monotone polygon into a chain of valid horizontal trapezoids.
// The trapezoid count is always smaller than the vertex count.
// Returns true if convertion succeeded.
split_into_trapezoids :: proc(poly: []Vec2, output: ^[dynamic]HorizEdge) -> bool {
	count := len(poly)
	if count < 3 {
		return false
	}

	// check if monotone and find indices of the highest and the lowest point
	top_index, bottom_index: int
	{
		// the idea is that a y-monotone polygon has only one local minimum

		less :: proc(poly: [^]Vec2, i, j: int) -> bool {
			return poly[i].y < poly[j].y || (poly[i].y == poly[j].y && i < j)
		}

		top_y, bottom_y := max(f32), min(f32)
		local_mins: int
		for i in 0 ..< count {
			// compare with the previous and the next
			ptr: [^]Vec2 = raw_data(poly)
			if less(ptr, i, (i + 1) % count) && less(ptr, i, (i - 1 + count) % count) {
				local_mins += 1
				if local_mins > 1 {
					return false
				}
			}

			y := poly[i].y
			if y < top_y {
				top_index = i
				top_y = y
			}
			if y > bottom_y {
				bottom_index = i
				bottom_y = y
			}
		}
		assert(local_mins == 1)
	}

	intersect :: proc(a: [2]Vec2, b: [2]Vec2) -> bool {
		EPS :: 1e-6

		r: Vec2 = a[1] - a[0]
		s: Vec2 = b[1] - b[0]
		qp: Vec2 = b[0] - a[0]
		rxs: f32 = cross2d(r, s)
		if -EPS < rxs && rxs < EPS {
			return false // zero, parallel
		}

		t: f32 = cross2d(qp, s) / rxs
		u: f32 = cross2d(qp, r) / rxs
		return EPS < t && t < 1 - EPS && EPS < u && u < 1 - EPS
	}

	// iterate from top to bottom, construct trapezoids,
	// abort if the polygon has self-intersections
	prev_len := len(output)
	bot: HorizEdge = {poly[top_index].x, poly[top_index].x, poly[top_index].y}
	b: int = top_index // backward
	f: int = top_index // forward
	for _ in 0 ..< count {
		bnext: int = (b - 1 + count) % count
		fnext: int = (f + 1) % count
		bseg: [2]Vec2 = {poly[b], poly[bnext]}
		fseg: [2]Vec2 = {poly[f], poly[fnext]}

		// check intersections
		if intersect(bseg, fseg) {
			resize(output, prev_len)
			return false
		}

		// skip horizontal edges
		if bseg[0].y == bseg[1].y {
			if bot.l == bseg[0].x {
				bot.l = bseg[1].x
			} else if bot.r == bseg[0].x {
				bot.r = bseg[1].x
			}
			if bot.l > bot.r {
				bot.l, bot.r = bot.r, bot.l
			}
			b = bnext
			continue
		}
		if fseg[0].y == fseg[1].y {
			if bot.l == fseg[0].x {
				bot.l = fseg[1].x
			} else if bot.r == fseg[0].x {
				bot.r = fseg[1].x
			}
			if bot.l > bot.r {
				bot.l, bot.r = bot.r, bot.l
			}
			f = fnext
			continue
		}

		// locate trapezoid bottom edge
		top: HorizEdge = bot
		if fequal2(fseg[1].y, bseg[1].y) // at the same height
		{
			bot.l = fseg[1].x
			bot.r = bseg[1].x
			bot.y = fseg[1].y
			f = fnext
			b = bnext
		} else if fseg[1].y < bseg[1].y {
			bot.r = fseg[1].x
			bot.y = fseg[1].y
			c: f32 = (bot.y - bseg[0].y) / (bseg[1].y - bseg[0].y)
			bot.l = bseg[0].x + c * (bseg[1].x - bseg[0].x)
			f = fnext
		} else {
			bot.l = bseg[1].x
			bot.y = bseg[1].y
			c: f32 = (bot.y - fseg[0].y) / (fseg[1].y - fseg[0].y)
			bot.r = fseg[0].x + c * (fseg[1].x - fseg[0].x)
			b = bnext
		}
		if bot.l > bot.r {
			bot.l, bot.r = bot.r, bot.l
		}

		if is_valid_trapezoid(top, bot) {
			if len(output) == prev_len || output[len(output) - 1] != top {
				append(output, top)
			}
			append(output, bot)
		}
	}
	return len(output) > prev_len
}

rasterize_trapezoid_chain :: proc(chain: []HorizEdge, params: RastParams, plotter: ^Plotter) {
	assert(len(chain) > 1)
	assert(params.clip.w > 0 && params.clip.h > 0)
	assert(plotter_has_all_methods(plotter))

	xmin, xmax := max(f32), min(f32)
	for &e in chain {
		xmin = min(xmin, e.l)
		xmax = max(xmax, e.r)
	}

	h_bounds := SpanI{max(params.clip.x, ifloor(xmin)), min(params.clip.x + params.clip.w, iceil(xmax))}
	if h_bounds.start >= h_bounds.end {
		return
	}
	v_bounds := SpanI{params.clip.y, params.clip.y + params.clip.h}
	accum := Accumulator {
		frame = h_bounds,
		width = h_bounds.end - h_bounds.start,
	}
	defer free(accum.scanline)

	y: int
	for i in 1 ..< len(chain) {
		top: HorizEdge = chain[i - 1]
		bot: HorizEdge = chain[i]
		assert(is_valid_trapezoid(top, bot))

		if fequal2(top.y, bot.y) {continue}
		if top.y >= f32(v_bounds.end) {break}
		if bot.y <= f32(v_bounds.start) {continue}

		itrap := TrapezoidI{top.l, top.r, bot.l, bot.r, 0, 0}

		height: f32 = bot.y - top.y
		step_l := (bot.l - top.l) / height
		step_r := (bot.r - top.r) / height
		top_bound := max(top.y, f32(v_bounds.start))
		bot_bound := min(bot.y, f32(v_bounds.end))
		y0: int = iround(top_bound)
		y1: int = iround(bot_bound)
		diff0, diff1: f32
		top_cap, bot_cap: bool

		if !fequal2(top.y, f32(y0)) {
			if f32(y0) < top.y {
				y0 += 1
			}
			diff0 = f32(y0) - top.y
			top_cap = diff0 < 1 && top_bound < f32(y0)
			itrap.tl += step_l * diff0
			itrap.tr += step_r * diff0
		}
		if !fequal2(bot.y, f32(y1)) {
			if bot.y < f32(y1) {
				y1 -= 1
			}
			diff1 = bot.y - f32(y1)
			bot_cap = diff1 < 1 && f32(y1) < bot_bound
			itrap.bl -= step_l * diff1
			itrap.br -= step_r * diff1
		}

		// a thin trapezoid inside one scan-line
		if y0 > y1 {
			y = y1
			accumulator_add(&accum, ScanLine{top.l, top.r, bot.l, bot.r}, height, step_l, step_r)
			continue
		}
		// top line
		if top_cap {
			accumulator_add(&accum, ScanLine{top.l, top.r, itrap.tl, itrap.tr}, diff0, step_l, step_r)
			if params.antialias {
				accumulator_plot_aa(&accum, y0 - 1, plotter)
			} else {
				accumulator_plot(&accum, y0 - 1, plotter)
			}
		}
		// the body within integer bounds
		if y0 < y1 {
			itrap.ty = y0
			itrap.by = y1
			if params.antialias {
				rasterize_trapezoid_i_aa(h_bounds, itrap, step_l, step_r, plotter)
			} else {
				rasterize_trapezoid_i(h_bounds, itrap, step_l, step_r, plotter)
			}
		}
		// bottom line
		if bot_cap {
			y = y1
			accumulator_add(&accum, ScanLine{itrap.bl, itrap.br, bot.l, bot.r}, diff1, step_l, step_r)
		}
	}

	if params.antialias {
		accumulator_plot_aa(&accum, y, plotter)
	} else {
		accumulator_plot(&accum, y, plotter)
	}
}

@(private)
SpanI :: struct {
	start, end: int,
}

@(private)
TrapezoidI :: struct {
	tl, tr: f32,
	bl, br: f32,
	ty, by: int,
}

@(private)
ScanLine :: struct {
	tl, tr: f32,
	bl, br: f32,
}

@(private)
Accumulator :: struct {
	// TODO: rewrite to store the difference in coverage
	scanline: [^]f32,
	frame:    SpanI,
	width:    int,
	ready:    bool,
}

@(private)
accumulator_initialize :: proc(acc: ^Accumulator) {
	if acc.scanline == nil {
		acc.scanline = raw_data(make([]f32, acc.width))
		assert(acc.scanline != nil)
	} else {
		for i in 0 ..< acc.width {
			acc.scanline[i] = 0
		}
	}
	acc.ready = true
}

@(private)
accumulator_add :: proc(acc: ^Accumulator, ln: ScanLine, height, step_l, step_r: f32) {
	if !acc.ready {
		accumulator_initialize(acc)
	}

	slope_l: f32 = step_l != 0 ? abs(1.0 / step_l) : 10.0
	slope_r: f32 = step_r != 0 ? abs(1.0 / step_r) : 10.0

	xf_ll := (step_l < 0 ? ln.bl : ln.tl) - f32(acc.frame.start)
	xf_lr := (step_l < 0 ? ln.tl : ln.bl) - f32(acc.frame.start)
	xf_rl := (step_r < 0 ? ln.br : ln.tr) - f32(acc.frame.start)
	xf_rr := (step_r < 0 ? ln.tr : ln.br) - f32(acc.frame.start)
	x_ll: int = ifloor(xf_ll)
	x_lr: int = iceil(xf_lr)
	x_rl: int = ifloor(xf_rl)
	x_rr: int = iceil(xf_rr)

	// the algorithm assumes that added lines never overlap,
	// which is true for trapezoid chain
	if x_ll < x_lr {
		len: int = x_lr - x_ll
		if slope_l / height <= 1 || len == 2 {
			next: int = x_ll + 1
			last: int = x_lr - 1
			b: f32
			{
				a0 := f32(next) - xf_ll
				b0 := a0 * slope_l
				b = b0
				accumulate(acc, x_ll, a0 * b0 / 2)
			}
			if len > 2 {
				area := (b + b + slope_l) / 2
				b += slope_l * f32(len - 2)
				for x in next ..< last {
					accumulate(acc, x, area)
					area += slope_l
				}
			}
			{
				a := xf_lr - f32(last)
				b = height - b
				accumulate(acc, last, a * (height - b / 2))
			}
		} else {
			assert(len == 1)
			accumulate(acc, x_ll, (min(xf_rl, f32(x_lr)) - xf_lr) * height)
		}
		// there may be a small rectangle between
		accumulate(acc, x_lr - 1, (min(xf_rl, f32(x_lr)) - xf_lr) * height)
	}

	for middle in max(x_lr, 0) ..< min(x_rl, acc.width) {
		acc.scanline[middle] += height
	}

	if x_rl < x_rr {
		accumulate(acc, x_rl, (xf_rl - max(xf_lr, f32(x_rl))) * height)

		len: int = x_rr - x_rl
		if slope_r / height <= 1 || len == 2 {
			next: int = x_rl + 1
			last: int = x_rr - 1
			b: f32
			{
				a0 := f32(next) - xf_rl
				b0 := a0 * slope_r
				b = height - b0
				accumulate(acc, x_rl, a0 * (height - b0 / 2))
			}
			if len > 2 {
				area := (b + b - slope_r) / 2
				b -= slope_r * f32(len - 2)
				for x in next ..< last {
					accumulate(acc, x, area)
					area -= slope_r
				}
			}
			{
				a := xf_rr - f32(last)
				accumulate(acc, last, a * b / 2)
			}
		} else {
			assert(len == 1)
			accumulate(acc, x_rl, (xf_rr - xf_rl) * height / 2)
		}
	}
}

@(private)
accumulate :: proc(acc: ^Accumulator, i: int, v: f32) {
	// super simple clipping
	if 0 <= i && i < acc.width {
		acc.scanline[i] += v
	}
}

@(private)
accumulator_plot_aa :: proc(acc: ^Accumulator, y: int, plotter: ^Plotter) {
	if !acc.ready {
		return
	}

	prev: int = acc.frame.start
	run := false
	for i in 0 ..< acc.width {
		cov := acc.scanline[i]
		if cov > 1 - EPS {
			if !run {
				prev = acc.frame.start + i
				run = true
			}
		} else {
			if run {
				plotter->set_scan_line(prev, acc.frame.start + i, y)
				run = false
			}
			if cov > EPS {
				plotter->mix_pixel(acc.frame.start + i, y, cov)
			}
		}
	}
	if run {
		plotter->set_scan_line(prev, acc.frame.end, y)
	}

	acc.ready = false
}

@(private)
accumulator_plot :: proc(acc: ^Accumulator, y: int, plotter: ^Plotter) {
	if !acc.ready {
		return
	}

	prev: int = acc.frame.start
	run := false
	for i in 0 ..< acc.width {
		cov := acc.scanline[i]
		if cov > 0.5 {
			if !run {
				prev = acc.frame.start + i
				run = true
			}
		} else if run {
			plotter->set_scan_line(prev, acc.frame.start + i, y)
			run = false
			break
		}
	}
	if run {
		plotter->set_scan_line(prev, acc.frame.end, y)
	}

	acc.ready = false
}

@(private)
rasterize_trapezoid_i_aa :: proc(clip: SpanI, trap: TrapezoidI, step_l, step_r: f32, plotter: ^Plotter) {
	slope_l: f32 = step_l != 0 ? abs(1.0 / step_l) : 10.0
	slope_r: f32 = step_r != 0 ? abs(1.0 / step_r) : 10.0

	ln := ScanLine{trap.tl, trap.tr, trap.tl, trap.tr}

	for y in trap.ty ..< trap.by {
		ln.bl += step_l
		ln.br += step_r

		// draw the scan line
		xf_ll: f32 = step_l < 0 ? ln.bl : ln.tl
		xf_lr: f32 = step_l < 0 ? ln.tl : ln.bl
		xf_rl: f32 = step_r < 0 ? ln.br : ln.tr
		xf_rr: f32 = step_r < 0 ? ln.tr : ln.br
		x_ll: int = ifloor(xf_ll)
		x_lr: int = iceil(xf_lr)
		x_rl: int = ifloor(xf_rl)
		x_rr: int = iceil(xf_rr)
		x_ll_ch: int = clamp(x_ll, clip.start, clip.end)
		x_lr_ch: int = clamp(x_lr, clip.start, clip.end)
		x_rl_ch: int = clamp(x_rl, clip.start, clip.end)
		x_rr_ch: int = clamp(x_rr, clip.start, clip.end)

		if x_ll_ch < x_lr_ch {
			len: int = x_lr - x_ll
			if slope_l <= 1 || len == 2 {
				next: int = x_ll + 1
				last: int = x_lr - 1
				b: f32
				{
					a0 := f32(next) - xf_ll
					b0 := a0 * slope_l
					b = b0
					if x_ll == x_ll_ch {
						area := a0 * b0 / 2
						plotter->mix_pixel(x_ll, y, area)
					}
				}
				for x in next ..< last {
					b0 := b
					b += slope_l
					if x_ll_ch <= x && x < x_lr_ch {
						area := (b0 + b) / 2
						plotter->mix_pixel(x, y, area)
					}
				}
				if x_lr == x_lr_ch {
					a := xf_lr - f32(last)
					b = 1 - b
					area := 1 - a * b / 2
					plotter->mix_pixel(last, y, area)
				}
			} else {
				assert(len == 1)
				area := f32(x_lr) - (xf_ll + xf_lr) / 2
				plotter->mix_pixel(x_ll, y, area)
			}
		}

		if x_lr_ch < x_rl_ch {
			plotter->set_scan_line(x_lr_ch, x_rl_ch, y)
		}

		if x_rl_ch < x_rr_ch {
			len: int = x_rr - x_rl
			if slope_r <= 1 || len == 2 {
				next: int = x_rl + 1
				last: int = x_rr - 1
				b: f32
				{
					a0 := f32(next) - xf_rl
					b0 := a0 * slope_r
					b = 1 - b0
					if x_rl == x_rl_ch {
						area := 1 - a0 * b0 / 2
						plotter->mix_pixel(x_rl, y, area)
					}
				}
				for x in next ..< last {
					b0 := b
					b -= slope_r
					if x_rl_ch <= x && x < x_rr_ch {
						area := (b0 + b) / 2
						plotter->mix_pixel(x, y, area)
					}
				}
				if x_rr == x_rr_ch {
					a := xf_rr - f32(last)
					area := a * b / 2
					plotter->mix_pixel(last, y, area)
				}
			} else {
				assert(len == 1)
				area := (xf_rl + xf_rr) / 2 - f32(x_rl)
				plotter->mix_pixel(x_rl, y, area)
			}
		}

		ln.tl = ln.bl
		ln.tr = ln.br
	}
}

@(private)
rasterize_trapezoid_i :: proc(clip: SpanI, trap: TrapezoidI, step_l, step_r: f32, plotter: ^Plotter) {
	trap := trap
	for y in trap.ty ..< trap.by {
		x0: int = max(iround(trap.tl), clip.start)
		x1: int = min(iround(trap.tr), clip.end)
		if x0 < x1 {
			plotter->set_scan_line(x0, x1, y)
		}
		trap.tl += step_l
		trap.tr += step_r
	}
}

// --- Line rasterizer ---

rasterize_line :: proc(p0, p1: Vec2, params: RastParams, plotter: ^Plotter) {
	assert(params.clip.w > 0 && params.clip.h > 0)
	assert(plotter_has_all_methods(plotter))

	p0, p1 := p0, p1
	b := params.clip

	// TODO: the line must not touch the right and bottom clip borders.
	// this is a quick fix for this issue, but not a correct one
	margin: f32 = (params.antialias ? 1.5 : 0.5) + EPS
	visible := clip_line(Rect{f32(b.x), f32(b.y), f32(b.x + b.w) - margin, f32(b.y + b.h) - margin}, &p0, &p1)
	if !visible {return}

	assert(p0.x >= 0 && p0.y >= 0 && p1.x >= 0 && p1.y >= 0)

	if params.antialias {
		dx := p1.x - p0.x
		dy := p1.y - p0.y
		ax := abs(dx)
		ay := abs(dy)
		if ax > ay {
			rasterize_line_hori_aa(p0.x, p0.y, p1.x, p1.y, ay < EPS, plotter)
		} else {
			rasterize_line_vert_aa(p0.x, p0.y, p1.x, p1.y, ax < EPS, plotter)
		}
	} else {
		x0i: int = iround(p0.x)
		y0i: int = iround(p0.y)
		x1i: int = iround(p1.x)
		y1i: int = iround(p1.y)
		rasterize_line_no_aa(x0i, y0i, x1i, y1i, plotter)
	}
}

@(private)
clip_line :: proc(clip: Rect, p0, p1: ^Vec2) -> bool {
	// Cohen–Sutherland clipping algorithm clips a line from
	// P0 = (x0, y0) to P1 = (x1, y1) against a rectangle with
	// diagonal from (left, top) to (right, bottom).
	// https://en.wikipedia.org/wiki/Cohen%E2%80%93Sutherland_algorithm

	Edge :: enum {
		left,
		right,
		bottom,
		top,
	}
	OutCode :: bit_set[Edge]

	// Compute the bit code for a point (x, y) using the clip rectangle
	// bounded diagonally by (left, top), and (right, bottom)
	compute_out_code :: proc(r: Rect, p: Vec2) -> OutCode {
		code: OutCode // initialised as being inside of clip window

		if p.x < r.left {
			code |= {.left}
		} else if p.x > r.right {
			code |= {.right}
		}
		if p.y < r.top {
			code |= {.top}
		} else if p.y > r.bottom {
			code |= {.bottom}
		}
		return code
	}

	// compute outcodes for P0, P1, and whatever point lies outside the clip rectangle
	outcode0 := compute_out_code(clip, p0^)
	outcode1 := compute_out_code(clip, p1^)
	accept := false

	for {
		if outcode0 | outcode1 == {} {
			// both points inside the window
			accept = true
			break
		}
		if outcode0 & outcode1 != {} {
			// both points outside the window
			break
		}

		// failed both tests, so calculate the line segment to clip
		// from an outside point to an intersection with clip edge
		p: Vec2

		// At least one endpoint is outside the clip rectangle; pick it.
		outcode_out := outcode0 != {} ? outcode0 : outcode1

		// Now find the intersection point;
		// use formulas y = y0 + slope * (x - x0), x = x0 + (1 / slope) * (y - y0)
		if .top in outcode_out { 	// point is above the clip rectangle
			p.x = p0.x + (p1.x - p0.x) * (clip.top - p0.y) / (p1.y - p0.y)
			p.y = clip.top
		} else if .bottom in outcode_out { 	// point is below the clip rectangle
			p.x = p0.x + (p1.x - p0.x) * (clip.bottom - p0.y) / (p1.y - p0.y)
			p.y = clip.bottom
		} else if .right in outcode_out { 	// point is to the right of clip rectangle
			p.y = p0.y + (p1.y - p0.y) * (clip.right - p0.x) / (p1.x - p0.x)
			p.x = clip.right
		} else if .left in outcode_out { 	// point is to the left of clip rectangle
			p.y = p0.y + (p1.y - p0.y) * (clip.left - p0.x) / (p1.x - p0.x)
			p.x = clip.left
		}

		// Now we move outside point to intersection point to clip
		// and get ready for next pass.
		if outcode_out == outcode0 {
			p0^ = p
			outcode0 = compute_out_code(clip, p0^)
		} else {
			p1^ = p
			outcode1 = compute_out_code(clip, p1^)
		}
	}
	return accept
}

@(private)
rasterize_line_hori_aa :: proc(x0, y0, x1, y1: f32, aligned: bool, plotter: ^Plotter) {
	x0, y0, x1, y1 := x0, y0, x1, y1

	dx := x1 - x0
	dy := y1 - y0
	if x0 > x1 {
		x0, x1 = x1, x0
		y0, y1 = y1, y0
	}
	gradient := dx != 0 ? dy / dx : 1

	// handle the first endpoint
	x_end := math.round(x0)
	y_end := y0 + gradient * (x_end - x0)
	x_gap := rfpart(x0 + 0.5)
	x0i := ipart(x_end)
	y0i := ipart(y_end)
	plotter->mix_pixel(x0i, y0i, rfpart(y_end) * x_gap)
	plotter->mix_pixel(x0i, y0i + 1, fpart(y_end) * x_gap)
	// first y-intersection for the main loop
	y_inter := y_end + gradient

	// handle the second endpoint
	x_end = math.round(x1)
	y_end = y1 + gradient * (x_end - x1)
	x_gap = fpart(x1 + 0.5)
	x1i := ipart(x_end)
	y1i := ipart(y_end)
	plotter->mix_pixel(x1i, y1i, rfpart(y_end) * x_gap)
	plotter->mix_pixel(x1i, y1i + 1, fpart(y_end) * x_gap)

	// main loop
	if aligned && x0i + 1 < x1i {
		y := math.round(y0)
		if fequal2(y, y0) {
			plotter->set_scan_line(x0i + 1, x1i, int(y))
		} else {
			top := math.floor(y0)
			topi := int(top)
			fract := y0 - top
			plotter->mix_scan_line(x0i + 1, x1i, topi, 1 - fract) // rfpart
			plotter->mix_scan_line(x0i + 1, x1i, topi + 1, fract) // fpart
		}
	} else {
		for x in x0i + 1 ..< x1i {
			y := ipart(y_inter)
			plotter->mix_pixel(x, y, f32(y + 1) - y_inter) // rfpart
			plotter->mix_pixel(x, y + 1, y_inter - f32(y)) // fpart
			y_inter += gradient
		}
	}
}

@(private)
rasterize_line_vert_aa :: proc(x0, y0, x1, y1: f32, aligned: bool, plotter: ^Plotter) {
	x0, y0, x1, y1 := x0, y0, x1, y1

	dx := x1 - x0
	dy := y1 - y0
	if y0 > y1 {
		x0, x1 = x1, x0
		y0, y1 = y1, y0
	}
	gradient := dy != 0 ? dx / dy : 1

	// handle the first endpoint
	y_end := math.round(y0)
	x_end := x0 + gradient * (y_end - y0)
	y_gap := rfpart(y0 + 0.5)
	y0i := ipart(y_end)
	x0i := ipart(x_end)
	plotter->mix_pixel(x0i, y0i, rfpart(x_end) * y_gap)
	plotter->mix_pixel(x0i + 1, y0i, fpart(x_end) * y_gap)
	// first x-intersection for the main loop
	x_inter := x_end + gradient

	// handle the second endpoint
	y_end = math.round(y1)
	x_end = x1 + gradient * (y_end - y1)
	y_gap = fpart(y1 + 0.5)
	y1i := ipart(y_end)
	x1i := ipart(x_end)
	plotter->mix_pixel(x1i, y1i, rfpart(x_end) * y_gap)
	plotter->mix_pixel(x1i + 1, y1i, fpart(x_end) * y_gap)

	// main loop
	x0rnd := math.round(x0)
	if aligned && y0i + 1 < y1i && fequal2(x0rnd, x0) {
		x := int(x0rnd)
		for y in y0i + 1 ..< y1i {
			plotter->set_pixel(x, y)
		}
	} else {
		for y in y0i + 1 ..< y1i {
			x := ipart(x_inter)
			plotter->mix_pixel(x, y, f32(x + 1) - x_inter) // rfpart
			plotter->mix_pixel(x + 1, y, x_inter - f32(x)) // fpart
			x_inter += gradient
		}
	}
}

@(private)
rasterize_line_no_aa :: proc(x0, y0, x1, y1: int, plotter: ^Plotter) {
	x0, y0, x1, y1 := x0, y0, x1, y1
	// fast path - horizontal
	if y0 == y1 {
		if x0 > x1 {
			x0, x1 = x1, x0
		}
		plotter->set_scan_line(x0, x1, y0)
		return
	}
	// fast path - vertical
	if x0 == x1 {
		if y0 > y1 {
			y0, y1 = y1, y0
		}
		for y in y0 ..< y1 {
			plotter->set_pixel(x0, y)
		}
		return
	}

	dx: int = x1 - x0
	ix: int = int(dx > 0) - int(dx < 0)
	dx2: int = abs(dx) * 2
	dy: int = y1 - y0
	iy: int = int(dy > 0) - int(dy < 0)
	dy2: int = abs(dy) * 2
	plotter->set_pixel(x0, y0)

	if dx2 >= dy2 {
		error: int = dy2 - dx2 / 2
		for x0 != x1 {
			if error >= 0 && (error != 0 || ix > 0) {
				error -= dx2
				y0 += iy
			}
			error += dy2
			x0 += ix
			plotter->set_pixel(x0, y0)
		}
	} else {
		error: int = dx2 - dy2 / 2
		for y0 != y1 {
			if error >= 0 && (error != 0 || iy > 0) {
				error -= dy2
				x0 += ix
			}
			error += dx2
			y0 += iy
			plotter->set_pixel(x0, y0)
		}
	}
}

// --- Utils ---

@(private)
fzero2 :: proc(a: f32) -> bool {return -0.00999999 < a && a < 0.00999999}
@(private)
fequal2 :: proc(a, b: f32) -> bool {return abs(a - b) < 0.0099999}

@(private)
ipart :: #force_inline proc(x: f32) -> int {return int(x)}
@(private)
ifloor :: #force_inline proc(x: f32) -> int {return int(math.floor(x))}
@(private)
iceil :: #force_inline proc(x: f32) -> int {return int(math.ceil(x))}
@(private)
iround :: #force_inline proc(x: f32) -> int {return int(math.round(x))}
@(private)
fpart :: #force_inline proc(x: f32) -> f32 {return x - math.floor(x)}
@(private)
rfpart :: #force_inline proc(x: f32) -> f32 {return math.floor(x) + 1 - x}

@(private)
cross2d :: proc(a, b: Vec2) -> f32 {return a.x * b.y - a.y * b.x}
