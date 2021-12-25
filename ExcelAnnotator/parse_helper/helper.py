def color2hex_rgb(color):
    """Transforms a color tuple (r,g,b) to rgb hex representation"""

    return (color[0] << 16) + (color[1] << 8) + color[2]


def color2hex_bgr(color):
    """Transforms a color tuple (r,g,b) to bgr hex representation"""

    return (color[2] << 16) + (color[1] << 8) + color[0]


def all_color():
    """Generator for colors. Used to mark position of cell inside of table."""

    for b in range(256):
        for g in range(256):
            for r in range(256):
                yield r, g, b

    # If all colors are used, we run into problems. Can be solved by introducing 'instances'.
    print("ERROR: Maximal color reached.")
    exit(1)


color_iter = all_color()


def next_fill_color(illegal_colors):
    """Get next valid color, ignores illegal colors which have been present in the document before."""

    fill_color = next(color_iter)
    while fill_color in illegal_colors:
        fill_color = next(color_iter)

    return fill_color
