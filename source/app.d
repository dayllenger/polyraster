module app;

// This is a usage demo for my rasterizer.
// It draws some polyspirals and writes the result as y4m video file.
// Note that y4m has lower quality than plain 32 bit images.

// $ dub run -c demo && vlc example.y4m

import polyraster;

enum {
    azure = Color(0xF0, 0xFF, 0xFF),
    dark_violet = Color(0x94, 0x00, 0xD3, 0xAA),
    turquoise = Color(0x40, 0xE0, 0xD0),

    W = 600,
    H = 600,
    L = 100,
    M = 85,
    FrameCount = 2,
}

void main()
{
    import std.array : appender;
    import std.file : write;

    auto buffer = appender!(ubyte[]);
    auto writer = Y4M_Writer(W, H, 1, 5, false, &buffer.put!ubyte);

    auto painter = new Painter(W, H);
    paint(painter);

    foreach (_; 0 .. FrameCount)
        writer.addFrame(painter.surface.data[0 .. W * H]);

    write("example.y4m", buffer.data);
}

void paint(Painter painter)
{
    Vec2[] poly = new Vec2[L + M];
    const uint[2] contours = [L, M];

    makePolySpiral(poly[0 .. L], 90);
    makePolySpiral(poly[L .. $], 121);

    painter.fillRect(10, 10, 580, 580, azure);
    painter.fillPoly(poly, contours, turquoise);
    painter.strokePoly(poly, contours, dark_violet);
}

void makePolySpiral(Vec2[] output, float angleIncrement)
{
    import std.math : cos, sin, PI;

    angleIncrement = angleIncrement * PI / 180;

    int len;
    float x = W / 2;
    float y = H / 2;
    float angle = angleIncrement;

    foreach (i; 0 .. output.length)
    {
        x += cos(angle) * len;
        y -= sin(angle) * len;
        output[i] = Vec2(x, y);

        len += 5;

        angle = (angle + angleIncrement) % (PI * 2);
    }
}

//===============================================================

align(1) struct Color
{
    ubyte r, g, b, a;

    void blend(Color src, uint alpha) nothrow @nogc
    {
        const uint ialpha = 255 - alpha;
        r = cast(ubyte)(((src.r * ialpha + r * alpha) >> 8) & 0xFF);
        g = cast(ubyte)(((src.g * ialpha + g * alpha) >> 8) & 0xFF);
        b = cast(ubyte)(((src.b * ialpha + b * alpha) >> 8) & 0xFF);
    }
}

struct ImageBuffer
{
    uint width, height;
    Color[] data;
}

// the demo painter can draw rectangles, polygons with several contours,
// and 1px polygon outline, filling them with opaque or transparent solid color

final class Painter
{
    ImageBuffer surface;
    private PlotterSolidOpaque plotter_op;
    private PlotterSolidTranslucent plotter_tr;

    this(uint width, uint height)
    {
        surface = ImageBuffer(width, height, new Color[width * height]);
        plotter_op = new PlotterSolidOpaque;
        plotter_tr = new PlotterSolidTranslucent;
    }

    private RastParams makeStdParams()
    {
        return RastParams(true, BoxI(0, 0, surface.width, surface.height), RastFillRule.odd);
    }

    private Plotter choosePlotter(Color c)
    {
        if (c.a == 0)
        {
            plotter_op.initialize(surface, c);
            return plotter_op;
        }
        {
            plotter_tr.initialize(surface, c);
            return plotter_tr;
        }
    }

    void fillRect(float x, float y, float w, float h, Color color)
    {
        const HorizEdge[2] t = [
            {x, x + w, y},
            {x, x + w, y + h},
        ];

        const RastParams params = makeStdParams();
        auto plotter = choosePlotter(color);
        rasterizeTrapezoidChain(t[], params, plotter);
    }

    void fillPoly(const Vec2[] points, const uint[] contours, Color color)
    {
        const RastParams params = makeStdParams();
        auto plotter = choosePlotter(color);
        rasterizePolygons(points, contours, params, plotter);
    }

    void strokePoly(const Vec2[] points, const uint[] contours, Color color)
    {
        const RastParams params = makeStdParams();
        auto plotter = choosePlotter(color);

        uint from;
        foreach (len; contours)
        {
            const uint to = from + len;
            foreach (i; from + 1 .. to)
            {
                const Vec2 p = points[i - 1];
                const Vec2 q = points[i];
                rasterizeLine(p, q, params, plotter);
            }
            {
                const Vec2 p = points[to - 1];
                const Vec2 q = points[from];
                rasterizeLine(p, q, params, plotter);
            }
            from = to;
        }
    }
}

final class PlotterSolidOpaque : Plotter
{
    Color* image;
    uint stride;
    Color color;

    void initialize(ref ImageBuffer surface, Color c)
    {
        image = surface.data.ptr;
        stride = surface.width;
        color = c;
    }

    void setPixel(int x, int y)
    {
        Color* pixel = image + y * stride + x;
        *pixel = color;
    }

    void mixPixel(int x, int y, float cov)
    {
        const a = 255 - cast(uint)(255 * cov);
        Color* pixel = image + y * stride + x;
        pixel.blend(color, a);
    }

    void setScanLine(int x0, int x1, int y)
    {
        Color* pixel = image + y * stride;
        pixel[x0 .. x1] = color;
    }

    void mixScanLine(int x0, int x1, int y, float cov)
    {
        const a = 255 - cast(uint)(255 * cov);
        Color* pixel = image + y * stride + x0;
        foreach (_; x0 .. x1)
        {
            pixel.blend(color, a);
            pixel++;
        }
    }
}

final class PlotterSolidTranslucent : Plotter
{
    Color* image;
    uint stride;
    Color color;
    float invAlpha;

    void initialize(ref ImageBuffer surface, Color c)
    {
        image = surface.data.ptr;
        stride = surface.width;
        color = c;
        invAlpha = 255 - c.a;
    }

    void setPixel(int x, int y)
    {
        Color* pixel = image + y * stride + x;
        pixel.blend(color, color.a);
    }

    void mixPixel(int x, int y, float cov)
    {
        const a = 255 - cast(uint)(invAlpha * cov);
        Color* pixel = image + y * stride + x;
        pixel.blend(color, a);
    }

    void setScanLine(int x0, int x1, int y)
    {
        Color* pixel = image + y * stride + x0;
        foreach (_; x0 .. x1)
        {
            pixel.blend(color, color.a);
            pixel++;
        }
    }

    void mixScanLine(int x0, int x1, int y, float cov)
    {
        const a = 255 - cast(uint)(invAlpha * cov);
        Color* pixel = image + y * stride + x0;
        foreach (_; x0 .. x1)
        {
            pixel.blend(color, a);
            pixel++;
        }
    }
}

//===============================================================

/// Writes y4m video files as either 444 or 444alpha.
struct Y4M_Writer
{
    import std.format : formattedWrite;
    import std.math : round;

    immutable
    {
        uint width, height;
        uint frameRateNumerator, frameRateDenominator;

        bool haveAlpha;

        void delegate(ubyte) output;
    }

    this(uint width, uint height, uint frameRateNumerator, uint frameRateDenominator, bool haveAlpha, void delegate(ubyte) output)
    in {
        assert(width > 0, "Width must be larger than 0");
        assert(height > 0, "Height must be larger than 0");
        assert(frameRateNumerator > 0, "Frame rate numerator must be larger than 0");
        assert(frameRateDenominator > 0, "Frame rate denominator must be larger than 0");
        assert(output !is null, "There must be a delegate to output bytes to");
    } do {
        this.width = width;
        this.height = height;
        this.frameRateNumerator = frameRateNumerator;
        this.frameRateDenominator = frameRateDenominator;
        this.haveAlpha = haveAlpha;
        this.output = output;

        printHeader();
    }

    private void printHeader()
    {
        output.formattedWrite!"YUV4MPEG2 W%d H%d F%d:%d Ip C%s\n"(
            width, height,
            frameRateNumerator, frameRateDenominator,
            haveAlpha ? "444alpha" : "444"
        );
    }

    /// Frame is RGB and compatible with Alpha
    void addFrame(const Color[] frame)
    {
        output.formattedWrite!"FRAME\n"();

        // Y
        foreach(c; frame)
            output(cast(ubyte)(0.257 * c.r + 0.504 * c.g + 0.098 * c.b + 16));

        // Cb
        foreach(c; frame)
            output(cast(ubyte)(-0.148 * c.r - 0.291 * c.g + 0.439 * c.b + 128));

        // Cr
        foreach(c; frame)
            output(cast(ubyte)(0.439 * c.r - 0.368 * c.g - 0.071 * c.b + 128));

        // A
        if (haveAlpha)
        {
            enum AlphaDelta = (16 - 235) / 256.0;

            foreach(c; frame)
                output(cast(ubyte)round(AlphaDelta * c.a + 16));
        }
    }
}
