from collections import namedtuple

# Render the tree in certain styles
Style = namedtuple("Style", ("vertical", "cont", "end"))
AsciiStyle = Style(vertical='|   ', cont='|-- ', end='+-- ')
ContStyle = Style(vertical='\u2502   ', cont='\u251c\u2500\u2500 ', end='\u2514\u2500\u2500 ')
ContRoundStyle = Style(vertical='\u2502   ', cont='\u251c\u2500\u2500 ', end='\u2570\u2500\u2500 ')
DoubleStyle = Style(vertical='\u2551   ', cont='\u2560\u2550\u2550 ', end='\u255a\u2550\u2550 ')
Row = namedtuple("Row", ("pre", "fill", "node"))