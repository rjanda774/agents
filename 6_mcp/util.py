from enum import Enum

css = """
.positive-pnl {
    color: green !important;
    font-weight: bold;
}
.positive-bg {
    background-color: green !important;
    font-weight: bold;
}
.negative-bg {
    background-color: red !important;
    font-weight: bold;
}
.negative-pnl {
    color: red !important;
    font-weight: bold;
}
.dataframe-fix-small .table-wrap {
min-height: 150px;
max-height: 150px;
}
.dataframe-fix .table-wrap {
min-height: 200px;
max-height: 200px;
}
/* Options positions table - compact all columns, small font */
#options-cathie table tbody tr,
#options-cathie table thead tr {
    height: 11px !important;
    line-height: 11px !important;
}
#options-cathie table td,
#options-cathie table th {
    font-size: 9px !important;
    padding: 0px 2px !important;
    height: 11px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    max-width: 52px !important;
}
/* Expiry col (5) - same width as Opened/Closed date cols */
#options-cathie table th:nth-child(5),
#options-cathie table td:nth-child(5) {
    max-width: 52px !important;
    min-width: 52px !important;
}
/* Closed col (8) - same width as Opened */
#options-cathie table th:nth-child(8),
#options-cathie table td:nth-child(8) {
    max-width: 52px !important;
    min-width: 52px !important;
}
footer{display:none !important}
"""


js = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""

class Color(Enum):
    RED = "#dd0000"
    GREEN = "#00dd00"
    YELLOW = "#dddd00"
    BLUE = "#0000ee"
    MAGENTA = "#aa00dd"
    CYAN = "#00dddd"
    WHITE = "#87CEEB"
