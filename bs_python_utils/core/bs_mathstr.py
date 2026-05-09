"""Some useful strings for math formulae.

Note:
    if the math looks strange in the documentation, just reload the page.

Attributes:
    * str_beta0: the LaTeX string $\\beta_0$
    * str_beta1: the LaTeX string $\\beta_1$
    * str_pi: the LaTeX string $\\pi$
    * str_sigma: the LaTeX string $\\sigma$
    * str_sigma2: the LaTeX string $\\sigma^2$
    * uni_beta0: the Unicode string $\\beta_0$
    * uni_beta1: the Unicode string $\\beta_1$
    * uni_pi: the Unicode string $\\pi$
    * uni_sigma: the Unicode string $\\sigma$
    * uni_sigma2: the Unicode string $\\sigma^2$
    * uni_s2: the Unicode string $s^2$
    * uni_R2: the Unicode string $R^2$
    * sub_sub_scripts: a dictionary of unicodes for subscripts and superscripts;
        e.g $a^b$ would be `"a" + sub_sup_scripts['b'][0]`
"""

#  LaTeX strings
str_beta0 = r"$\beta_0$"
str_beta1 = r"$\beta_1$"
str_pi = r"$\pi$"
str_sigma = r"$\sigma$"
str_sigma2 = r"$\sigma^2$"

# * unicode
uni_beta0 = "\N{GREEK SMALL LETTER BETA}\N{SUBSCRIPT ZERO}"
uni_beta1 = "\N{GREEK SMALL LETTER BETA}\N{SUBSCRIPT ONE}"
uni_pi = "\N{GREEK SMALL LETTER PI}"
uni_sigma = "\N{GREEK SMALL LETTER SIGMA}"
uni_sigma2 = "\N{GREEK SMALL LETTER SIGMA}\N{SUPERSCRIPT TWO}"
uni_s2 = "s\N{SUPERSCRIPT TWO}"
uni_R2 = "R\N{SUPERSCRIPT TWO}"


sub_sup_scripts = {
    #           superscript     subscript
    "0": ("\u2070", "\u2080"),
    "1": ("\u00b9", "\u2081"),
    "2": ("\u00b2", "\u2082"),
    "3": ("\u00b3", "\u2083"),
    "4": ("\u2074", "\u2084"),
    "5": ("\u2075", "\u2085"),
    "6": ("\u2076", "\u2086"),
    "7": ("\u2077", "\u2087"),
    "8": ("\u2078", "\u2088"),
    "9": ("\u2079", "\u2089"),
    "a": ("\u1d43", "\u2090"),
    "b": ("\u1d47", "?"),
    "c": ("\u1d9c", "?"),
    "d": ("\u1d48", "?"),
    "e": ("\u1d49", "\u2091"),
    "f": ("\u1da0", "?"),
    "g": ("\u1d4d", "?"),
    "h": ("\u02b0", "\u2095"),
    "i": ("\u2071", "\u1d62"),
    "j": ("\u02b2", "\u2c7c"),
    "k": ("\u1d4f", "\u2096"),
    "l": ("\u02e1", "\u2097"),
    "m": ("\u1d50", "\u2098"),
    "n": ("\u207f", "\u2099"),
    "o": ("\u1d52", "\u2092"),
    "p": ("\u1d56", "\u209a"),
    "q": ("?", "?"),
    "r": ("\u02b3", "\u1d63"),
    "s": ("\u02e2", "\u209b"),
    "t": ("\u1d57", "\u209c"),
    "u": ("\u1d58", "\u1d64"),
    "v": ("\u1d5b", "\u1d65"),
    "w": ("\u02b7", "?"),
    "x": ("\u02e3", "\u2093"),
    "y": ("\u02b8", "?"),
    "z": ("?", "?"),
    "A": ("\u1d2c", "?"),
    "B": ("\u1d2e", "?"),
    "C": ("?", "?"),
    "D": ("\u1d30", "?"),
    "E": ("\u1d31", "?"),
    "F": ("?", "?"),
    "G": ("\u1d33", "?"),
    "H": ("\u1d34", "?"),
    "I": ("\u1d35", "?"),
    "J": ("\u1d36", "?"),
    "K": ("\u1d37", "?"),
    "L": ("\u1d38", "?"),
    "M": ("\u1d39", "?"),
    "N": ("\u1d3a", "?"),
    "O": ("\u1d3c", "?"),
    "P": ("\u1d3e", "?"),
    "Q": ("?", "?"),
    "R": ("\u1d3f", "?"),
    "S": ("?", "?"),
    "T": ("\u1d40", "?"),
    "U": ("\u1d41", "?"),
    "V": ("\u2c7d", "?"),
    "W": ("\u1d42", "?"),
    "X": ("?", "?"),
    "Y": ("?", "?"),
    "Z": ("?", "?"),
    "+": ("\u207a", "\u208a"),
    "-": ("\u207b", "\u208b"),
    "=": ("\u207c", "\u208c"),
    "(": ("\u207d", "\u208d"),
    ")": ("\u207e", "\u208e"),
    ":alpha": ("\u1d45", "?"),
    ":beta": ("\u1d5d", "\u1d66"),
    ":gamma": ("\u1d5e", "\u1d67"),
    ":delta": ("\u1d5f", "?"),
    ":epsilon": ("\u1d4b", "?"),
    ":theta": ("\u1dbf", "?"),
    ":iota": ("\u1da5", "?"),
    ":pho": ("?", "\u1d68"),
    ":phi": ("\u1db2", "?"),
    ":psi": ("\u1d60", "\u1d69"),
    ":chi": ("\u1d61", "\u1d6a"),
    ":coffee": ("\u2615", "\u2615"),
}
